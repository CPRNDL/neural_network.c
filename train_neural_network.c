// MNIST 숫자 인식 인공신경망 (개선 버전)
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

#define TRAIN_DATA 1000   // 작은 데이터셋으로 테스트
#define TEST_DATA 100
#define INPUT 784
#define HIDDEN 30
#define OUTPUT 10
#define ETA 0.01
#define EPOCH 10
#define BATCH_SIZE 100  // 미니배치 크기

// 동적 메모리 할당을 위한 포인터들
double **x_train;
int *y_train;
double **x_test;
int *y_test;

// 신경망 파라미터
double w_ih[HIDDEN][INPUT];
double w_ho[OUTPUT][HIDDEN];
double b_h[HIDDEN];
double b_o[OUTPUT];

// --- [추가] 미니배치 그래디언트 누적을 위한 변수 ---
double grad_w_ih[HIDDEN][INPUT];
double grad_w_ho[OUTPUT][HIDDEN];
double grad_b_h[HIDDEN];
double grad_b_o[OUTPUT];

// 중간값 저장용
double a_h[HIDDEN];
double a_o[OUTPUT];
double d_h[HIDDEN];
double d_o[OUTPUT];

// 메모리 할당 함수
int allocate_memory() {
    // 훈련 데이터 메모리 할당
    x_train = (double**)malloc(TRAIN_DATA * sizeof(double*));
    if (!x_train) return 0;
    for (int i = 0; i < TRAIN_DATA; i++) {
        x_train[i] = (double*)malloc(INPUT * sizeof(double));
        if (!x_train[i]) return 0;
    }
    y_train = (int*)malloc(TRAIN_DATA * sizeof(int));
    if (!y_train) return 0;

    // 테스트 데이터 메모리 할당
    x_test = (double**)malloc(TEST_DATA * sizeof(double*));
    if (!x_test) return 0;
    for (int i = 0; i < TEST_DATA; i++) {
        x_test[i] = (double*)malloc(INPUT * sizeof(double));
        if (!x_test[i]) return 0;
    }
    y_test = (int*)malloc(TEST_DATA * sizeof(int));
    if (!y_test) return 0;

    return 1;
}

// 메모리 해제 함수
void free_memory() {
    if (x_train) {
        for (int i = 0; i < TRAIN_DATA; i++) {
            if (x_train[i]) free(x_train[i]);
        }
        free(x_train);
    }
    if (y_train) free(y_train);
    if (x_test) {
        for (int i = 0; i < TEST_DATA; i++) {
            if (x_test[i]) free(x_test[i]);
        }
        free(x_test);
    }
    if (y_test) free(y_test);
}

// 시그모이드
double sigmoid(double x) {
    // 수치적 안정성을 위한 클리핑
    if (x > 500) return 1.0;
    if (x < -500) return 0.0;
    return 1.0 / (1.0 + exp(-x));
}

// 소프트맥스 (개선된 버전)
void softmax(double z[OUTPUT], double out[OUTPUT]) {
    double max = z[0];
    for (int i = 1; i < OUTPUT; i++) {
        if (z[i] > max) max = z[i];
    }
    
    double sum = 0.0;
    for (int i = 0; i < OUTPUT; i++) {
        out[i] = exp(z[i] - max);
        sum += out[i];
    }
    
    // 수치적 안정성 체크
    if (sum < 1e-10) sum = 1e-10;
    
    for (int i = 0; i < OUTPUT; i++) {
        out[i] /= sum;
        // 최소값 보장
        if (out[i] < 1e-10) out[i] = 1e-10;
    }
}

// Xavier 초기화
double xavier_init(int fan_in, int fan_out) {
    double limit = sqrt(6.0 / (fan_in + fan_out));
    return ((double)rand() / RAND_MAX) * 2 * limit - limit;
}

// 개선된 데이터 로딩 함수
int load_data(const char* filename, double **x, int *y, int count) {
    FILE* fp = fopen(filename, "r");
    if (!fp) {
        printf("Cannot open file: %s\n", filename);
        return 0;
    }
    
    printf("Loading data from: %s\n", filename);
    
    // 파일 크기 확인
    fseek(fp, 0, SEEK_END);
    long file_size = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    printf("File size: %ld bytes\n", file_size);
    
    // 첫 번째 줄 읽기 테스트
    char first_line[8000];  // 충분히 큰 버퍼
    if (fgets(first_line, sizeof(first_line), fp) != NULL) {
        printf("First line preview (first 100 chars): %.100s\n", first_line);
        printf("First line length: %d\n", (int)strlen(first_line));
    }
    
    // 파일 포인터를 처음으로 되돌리기
    fseek(fp, 0, SEEK_SET);
    
    for (int i = 0; i < count; i++) {
        // 레이블 읽기
        int label_result = fscanf(fp, "%d", &y[i]);
        if (label_result != 1) {
            printf("Label reading error at line %d\n", i);
            printf("fscanf returned: %d\n", label_result);
            
            // 현재 파일 위치 확인
            long pos = ftell(fp);
            printf("Current file position: %ld\n", pos);
            
            // 다음 몇 글자 확인
            char buffer[100];
            if (fgets(buffer, sizeof(buffer), fp) != NULL) {
                printf("Next characters: %.50s\n", buffer);
            }
            
            fclose(fp);
            return 0;
        }
        
        // 레이블 값 확인
        if (y[i] < 0 || y[i] > 9) {
            printf("Invalid label %d at line %d\n", y[i], i);
            fclose(fp);
            return 0;
        }
        
        // 픽셀 데이터 읽기
        for (int j = 0; j < INPUT; j++) {
            int pixel_value;
            int pixel_result = fscanf(fp, "%d", &pixel_value);
            if (pixel_result != 1) {
                printf("Pixel data reading error at line %d, column %d\n", i, j);
                printf("fscanf returned: %d\n", pixel_result);
                
                // 현재 파일 위치와 다음 문자들 확인
                long pos = ftell(fp);
                printf("Current file position: %ld\n", pos);
                
                char buffer[50];
                if (fgets(buffer, sizeof(buffer), fp) != NULL) {
                    printf("Next characters: %.30s\n", buffer);
                }
                
                fclose(fp);
                return 0;
            }
            
            // 픽셀 값 범위 확인
            if (pixel_value < 0 || pixel_value > 255) {
                printf("Invalid pixel value %d at line %d, column %d\n", pixel_value, i, j);
                fclose(fp);
                return 0;
            }
            
            x[i][j] = pixel_value / 255.0;  // 정규화
        }
        
        // 첫 번째 샘플 정보 출력
        if (i == 0) {
            printf("First sample - Label: %d, First few pixels: %.3f %.3f %.3f\n", 
                   y[i], x[i][0], x[i][1], x[i][2]);
        }
        
        // 진행 상황 출력
        if ((i + 1) % 100 == 0) {
            printf("  %d/%d loaded\n", i + 1, count);
        }
    }
    
    fclose(fp);
    printf("Data loading complete: %d samples\n", count);
    return 1;
}

// 파라미터 초기화
void init_parameters() {
    printf("Initializing neural network parameters...\n");
    
    // 입력-은닉층 가중치 초기화
    for (int i = 0; i < HIDDEN; i++) {
        b_h[i] = 0.0;  // 편향은 0으로 초기화
        for (int j = 0; j < INPUT; j++) {
            w_ih[i][j] = xavier_init(INPUT, HIDDEN);
        }
    }
    
    // 은닉-출력층 가중치 초기화
    for (int i = 0; i < OUTPUT; i++) {
        b_o[i] = 0.0;  // 편향은 0으로 초기화
        for (int j = 0; j < HIDDEN; j++) {
            w_ho[i][j] = xavier_init(HIDDEN, OUTPUT);
        }
    }
    
    printf("Initialization completed\n");
}

// 순전파
void forward(double input[INPUT]) {
    // 은닉층 계산
    for (int i = 0; i < HIDDEN; i++) {
        double sum = b_h[i];
        for (int j = 0; j < INPUT; j++) {
            sum += w_ih[i][j] * input[j];
        }
        a_h[i] = sigmoid(sum);
    }
    
    // 출력층 계산
    double z[OUTPUT];
    for (int i = 0; i < OUTPUT; i++) {
        z[i] = b_o[i];
        for (int j = 0; j < HIDDEN; j++) {
            z[i] += w_ho[i][j] * a_h[j];
        }
    }
    
    softmax(z, a_o);
}

// 역전파 (수정된 버전)
void backward(double input[INPUT], int label) {
    // 출력층 오차 계산 (이전과 동일)
    for (int i = 0; i < OUTPUT; i++) {
        d_o[i] = a_o[i] - (i == label ? 1.0 : 0.0);
    }
    
    // 은닉층 오차 계산 (이전과 동일)
    for (int i = 0; i < HIDDEN; i++) {
        double sum = 0.0;
        for (int j = 0; j < OUTPUT; j++) {
            sum += d_o[j] * w_ho[j][i];
        }
        d_h[i] = sum * a_h[i] * (1.0 - a_h[i]);
    }
    
    // --- [변경] 가중치 업데이트 -> 그래디언트 누적 ---
    // 은닉-출력층 그래디언트 누적
    for (int i = 0; i < OUTPUT; i++) {
        for (int j = 0; j < HIDDEN; j++) {
            grad_w_ho[i][j] += d_o[i] * a_h[j]; // 누적
        }
        grad_b_o[i] += d_o[i]; // 누적
    }
    
    // 입력-은닉층 그래디언트 누적
    for (int i = 0; i < HIDDEN; i++) {
        for (int j = 0; j < INPUT; j++) {
            grad_w_ih[i][j] += d_h[i] * input[j]; // 누적
        }
        grad_b_h[i] += d_h[i]; // 누적
    }
}

// --- [추가] 가중치 업데이트 함수 ---
void update_parameters() {
    // 은닉-출력층 가중치 및 편향 업데이트
    for (int i = 0; i < OUTPUT; i++) {
        for (int j = 0; j < HIDDEN; j++) {
            w_ho[i][j] -= ETA * grad_w_ho[i][j] / BATCH_SIZE;
        }
        b_o[i] -= ETA * grad_b_o[i] / BATCH_SIZE;
    }
    
    // 입력-은닉층 가중치 및 편향 업데이트
    for (int i = 0; i < HIDDEN; i++) {
        for (int j = 0; j < INPUT; j++) {
            w_ih[i][j] -= ETA * grad_w_ih[i][j] / BATCH_SIZE;
        }
        b_h[i] -= ETA * grad_b_h[i] / BATCH_SIZE;
    }

    // 그래디언트 누적 변수 초기화
    memset(grad_w_ih, 0, sizeof(grad_w_ih));
    memset(grad_w_ho, 0, sizeof(grad_w_ho));
    memset(grad_b_h, 0, sizeof(grad_b_h));
    memset(grad_b_o, 0, sizeof(grad_b_o));
}

// 예측
int predict(double input[INPUT]) {
    forward(input);
    int max_index = 0;
    for (int i = 1; i < OUTPUT; i++) {
        if (a_o[i] > a_o[max_index]) {
            max_index = i;
        }
    }
    return max_index;
}

// 개선된 데이터 섞기
void shuffle_data() {
    for (int i = TRAIN_DATA - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        
        // x_train 교환
        double *temp_x = x_train[i];
        x_train[i] = x_train[j];
        x_train[j] = temp_x;
        
        // y_train 교환
        int temp_y = y_train[i];
        y_train[i] = y_train[j];
        y_train[j] = temp_y;
    }
}

// 정확도 계산
double calculate_accuracy(double **x, int *y, int count) {
    int correct = 0;
    for (int i = 0; i < count; i++) {
        int pred = predict(x[i]);
        if (pred == y[i]) correct++;
    }
    return 100.0 * correct / count;
}

int main() {
    printf("MNIST Neural Network Started\n");
    printf("Architecture: %d-%d-%d, Learning Rate: %.3f, Epochs: %d\n", 
           INPUT, HIDDEN, OUTPUT, ETA, EPOCH);
    
    srand((unsigned int)time(NULL));
    
    // 메모리 할당
    if (!allocate_memory()) {
        printf("Memory allocation failed\n");
        return -1;
    }
    
    // 데이터 파일 존재 확인
    FILE* test_file = fopen("train.txt", "r");
    if (!test_file) {
        printf("Error: train.txt file not found!\n");
        printf("Please make sure you have:\n");
        printf("  - train.txt (60000 samples)\n");
        printf("  - test.txt (1000 samples)\n");
        printf("Format: label pixel1 pixel2 ... pixel784\n");
        free_memory();
        return -1;
    }
    fclose(test_file);
    
    test_file = fopen("test.txt", "r");
    if (!test_file) {
        printf("Error: test.txt file not found!\n");
        free_memory();
        return -1;
    }
    fclose(test_file);
    
    // 데이터 로딩
    if (!load_data("train.txt", x_train, y_train, TRAIN_DATA)) {
        printf("Training data loading failed\n");
        free_memory();
        return -1;
    }
    
    if (!load_data("test.txt", x_test, y_test, TEST_DATA)) {
        printf("Test data loading failed\n");
        free_memory();
        return -1;
    }
    
    // 신경망 초기화
    init_parameters();
    
    printf("\nTraining started...\n");
    printf("===========================================\n");
    
    // 학습 루프
    for (int epoch = 0; epoch < EPOCH; epoch++) {
    shuffle_data();
    double total_cost = 0.0;
    
    // --- [변경] 미니배치 학습 루프 ---
    for (int i = 0; i < TRAIN_DATA; i += BATCH_SIZE) {
        // 그래디언트 누적 변수 초기화 (매 배치 시작 전)
        memset(grad_w_ih, 0, sizeof(grad_w_ih));
        memset(grad_w_ho, 0, sizeof(grad_w_ho));
        memset(grad_b_h, 0, sizeof(grad_b_h));
        memset(grad_b_o, 0, sizeof(grad_b_o));

        // BATCH_SIZE 만큼 순전파 및 역전파(그래디언트 누적) 수행
        for (int j = 0; j < BATCH_SIZE; j++) {
            int data_index = i + j;
            if (data_index >= TRAIN_DATA) break;

            forward(x_train[data_index]);
            backward(x_train[data_index], y_train[data_index]);
            
            // Cross-entropy 손실 계산
            double cost = -log(a_o[y_train[data_index]] + 1e-15);
            total_cost += cost;
        }

        // 배치가 끝나면 누적된 그래디언트로 파라미터 업데이트
        update_parameters(); 
    }
    
    double avg_cost = total_cost / TRAIN_DATA;
    double train_acc = calculate_accuracy(x_train, y_train, 1000);
    double test_acc = calculate_accuracy(x_test, y_test, TEST_DATA);
    
    printf("Epoch %2d | Cost: %.4f | Train Acc: %.2f%% | Test Acc: %.2f%%\n", 
           epoch + 1, avg_cost, train_acc, test_acc);
}
    
    // 메모리 해제
    free_memory();
    
    return 0;
}