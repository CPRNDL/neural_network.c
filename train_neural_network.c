/*
프로그래밍 C반 수행평가
주제: 28 * 28 필기체 숫자 인식 인공 신경망 구현
30509민재헌
*/


//라이브러리 include
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>


//상수 매크로 선언
#define TRAIN_DATA 10000 //훈련 데이터 수
#define TEST_DATA 1000   //테스트 데이터 수
#define INPUT 784        //데이터 크기 -> 784 = 28 * 28
#define HIDDEN 30        //은닉층 노드 수
#define OUTPUT 10        //출력층 노드 수 -> 출력층은 0~1 사이의 값을 가지기 때문에 10개로 설정
#define ETA 0.05         //학습률(η)
#define EPOCH 10         //학습 횟수


//전역 변수 선언
double x_train[TRAIN_DATA][INPUT];    //학습 데이터
int y_train[TRAIN_DATA];              //학습 정답 데이터
double x_test[TEST_DATA][INPUT];     //테스트 데이터
int y_test[TEST_DATA];               //테스트 정답 데이터

double w_ih[HIDDEN][INPUT];     //입력층 -> 은닉층 가중치
double w_ho[OUTPUT][HIDDEN];    //은닉층 -> 출력층 가중치
double b_h[HIDDEN];             //은닉층 편향
double b_o[OUTPUT];             //출력층 편향
double a_h[HIDDEN];             //은닉층의 시그모이드 값
double a_o[OUTPUT];             //출력층의 시그모이드 값
double d_h[HIDDEN];             //은닉층의 오차
double d_o[OUTPUT];             //출력층의 오차


//활성화 함수: 시그모이드 함수
double sigmoid(double z){
    return 1/(1+exp(-z));   //시그모이드 함숫값을 반환
}


//초기 가중치, 편향 설정을 위한 난수 생성 함수: Xavier Initialization
double random(int n) {
    double limit = sqrt(1.0/n);                         //역전파 기울기 소실과 폭주를 막기 위해 노드 수가 많을수록 범위값이 줄어들도록 설정
    return ((double)rand()/RAND_MAX)*2*limit-limit;     //rand() 함수는 0~RAND_MAX를 반환하므로 이를 RAND_MAX로 나누면 0~1 사이의 수가 됨
                                                        //*2*limit-limit -> -limit ~ +limit 범위로 변환
}


//데이터 파일 읽기 함수
void load_data(const char* filename, double x[][INPUT], int y[], int data){    //파일 이름이 배열의 포인터를 가리키므로 const char 포인터로 받음
    FILE *fp = fopen(filename, "r");                //fopen으로 파일 열기 -> return 주소 값을 FILE 구조체 포인터에 넣음
    if(fp==NULL){                                   //포인터가 NULL을 반환 -> 인식 못 함
        printf("Failed to open file\n");            //파일 열기 실패 출력
        exit(1);                                    //비정상 종료
    }
    for(int i=0; i<data; i++){                      //데이터 수 만큼 반복
        int temp;                                   //임시 변수
        if(fscanf(fp, "%d", &temp)==EOF){           //fscanf 실패시
            printf("Error in scanning label\n");    //오류 메시지 출력
            exit(1);                                //비정상 종료
        }
        y[i]=temp;                                  //정답 데이터 넣기
        for(int j=0; j<INPUT; j++){                 //28*28 만큼 반복
            if(fscanf(fp, "%d", &temp)==EOF){       //fscanf 실패시
                printf("Error in scanning data\n"); //오류 메시지 출력
                exit(1);                            //비정상 종료
            }
            x[i][j]=(double)temp/255;               //데이터 넣기
        }
    }
    fclose(fp);     //파일 닫기
}


//가중치 편향 초기화 함수
void init_parameters(){
    for(int i=0; i<HIDDEN; i++){
        b_h[i]=random(INPUT);           //은닉층 편향을 무작위로 설정
        for(int j=0; j<INPUT; j++){
            w_ih[i][j]=random(INPUT);   //입력층->은닉층 가중치를 무작위로 설정
        }
    }
    for(int i=0; i<OUTPUT; i++){
        b_o[i]=random(HIDDEN);          //출력층 편향을 무작위로 설정
        for(int j=0; j<HIDDEN; j++){
            w_ho[i][j]=random(HIDDEN);  //은닉층->출력층 가중치를 무작위로 설정
        }
    }
}


//데이터 학습 루프 함수들
//순전파
void forward(int index){                        //k번째 데이터 인덱스를 받음
    double sum;                                 //누적합 선언
    for(int i=0; i<HIDDEN; i++){
        sum=b_h[i];                             //누적합에 은닉층 편향 더함
        for(int j=0; j<INPUT; j++){
            sum+=w_ih[i][j]*x_train[index][j];  //누적합에 은닉층 가중 입력값을 더함
        }
        a_h[i]=sigmoid(sum);                    //은닉층에 누적합의 활성화 함숫값을 대입
    }

    for(int i=0; i<OUTPUT; i++){
        sum=b_o[i];                             //누적합에 출력층 편향 더함
        for(int j=0; j<HIDDEN; j++){
            sum+=w_ho[i][j]*a_h[j];             //누적합에 출력층 가중 입력값을 더함
        }
        a_o[i]=sigmoid(sum);                    //출력층에 누적합의 활성화 함숫값을 대입
    }
}

//오차역전파
double backward(int label){                     //정답 레이블을 받음
    double C=0.0;                               //비용 함숫값 선언
    for(int i=0; i<OUTPUT; i++){
        double t = (i == label) ? 1.0 : 0.0;    //
        double y = a_o[i];                      //
        C += 0.5 * pow(t - y, 2);               //비용 함숫값에 레이블과 출력층값의 차를 제곱하고 1/2을 곱한 값을 더함
        d_o[i] = (t - y) * y * (1 - y);         //출력층의 오차값(비용 함수의 출력층 편미분값)
    }
    for(int i=0; i<HIDDEN; i++){
        double sum = 0.0;                       //누적합 선언
        for(int j=0; j<OUTPUT; j++){
            sum += d_o[j] * w_ho[j][i];         //누적합에 출력층 오차와 은닉층 -> 출력층 가중치를 곱한 값을 더함
        }
        double h = a_h[i];                      //출력층 가중 입력값
        d_h[i] = sum * h * (1 - h);             //은닉층의 오차값(비용 함수의 가중치 편미분값, 비용 함수의 편향 편미분값)
    }
    return C;                                   //최종 비용 함숫값 return
}

//파라미터 갱신 함수
void update_parameters(int index){                      //k번째 데이터 인덱스를 받음
    for(int i=0; i<OUTPUT; i++){
        for(int j=0; j<HIDDEN; j++){
            w_ho[i][j]-=ETA*d_o[i]*a_h[j];              //경사하강법으로 은닉층 -> 출력층 가중치 갱신
        }
        b_o[i]-=ETA*d_o[i];                             //출력층 편향 갱신
    }
    for(int i=0; i<HIDDEN; i++){
        for(int j=0; j<INPUT; j++){
            w_ih[i][j]-=ETA*d_h[i]*x_train[index][j];   //입력층 -> 은닉층 가중치 갱신
        }
        b_h[i]-=ETA*d_h[i];                             //은닉층 편향 갱신
    }
}


//데이터 섞는 함수
void shuffle_data(double x[][INPUT], int y[], int data){
    srand((unsigned)time(NULL));

    for(int i=data-1; i>0; i--){
        int j = rand() % (i + 1);
        for(int k=0; k<INPUT; k++){
            double tempx = x[i][k];
            x[i][k] = x[j][k];
            x[j][k] = tempx;
        }
        int tempy = y[i];
        y[i] = y[j];
        y[j] = tempy;
    }
}


//예측 함수
void predict(double x[INPUT], double* result){
    double sum;
    for(int i=0; i<HIDDEN; i++){
        sum = b_h[i];
        for(int j=0; j<INPUT; j++){
            sum += w_ih[i][j] * x[j];
        }
        a_h[i] = sigmoid(sum);
    }

    for(int i=0; i<OUTPUT; i++){
        sum = b_o[i];
        for(int j=0; j<HIDDEN; j++)
            sum += w_ho[i][j] * a_h[j];
        a_o[i] = sigmoid(sum);
    }

    int M_index = 0;
    double M_value = a_o[0];
    for(int i=1; i<OUTPUT; i++){
        if(a_o[i] > M_value){
            M_value = a_o[i];
            M_index = i;
        }
    }
    *result = (double)M_index;
}


int main(){
    load_data("train.txt", x_train, y_train, TRAIN_DATA);   //훈련 데이터 읽기
    load_data("test.txt", x_test, y_test, TEST_DATA);      //테스트 데이터 읽기
    srand((unsigned int)time(NULL));            //시드 초기화

    init_parameters();                  //파라미터 초기화

    printf("Training begins...\n");
    for(int epoch=0; epoch<EPOCH; epoch++){
        double epoch_C=0.0;                     //epoch 비용 함숫값 선언
        shuffle_data(x_train, y_train, TRAIN_DATA);                         //데이터 섞기
        for(int i=0; i<TRAIN_DATA; i++){
            int label_index=y_train[i];         //k번째 데이터의 레이블
            forward(i);                         //순전파
            epoch_C+=backward(label_index);     //역전파, 비용 함숫값을 받음
            update_parameters(i);               //파라미터 갱신
        }
        printf("[Epoch %d] Cost: %lf\n", epoch+1, epoch_C/TRAIN_DATA);    //매 epoch마다 평균 비용 함숫값 출력
    }

    printf("Training complete...\nTesting begins...\n");

    int correct = 0;
    for(int i=0; i<TEST_DATA; i++){
        double pred;
        predict(x_test[i], &pred);
        if((int)pred == y_test[i]){
            correct++;
            printf("Correct\n");
        }else{
            printf("Incorrect\n");
        }
    }
    double acc = (double)correct / TEST_DATA * 100.0;
    printf("Accuracy: %.2lf%%\n", acc);

    printf("Testing complete...\n\n");
    double x_user[INPUT];
    printf("Please enter 784 values (0~255)\n");
    for(int i=0; i<INPUT; i++){
        int temp;
        if(scanf("%d", &temp) != 1){
            printf("Invalid input\nPlease enter 0~255 value\n");
            exit(1);
        }
        if(temp < 0) temp = 0;
        if(temp > 255) temp = 255;
        x_user[i] = temp / 255.0;
    }
    double user_result;
    predict(x_user, &user_result);
    printf("The number is %d\n", (int)user_result);
}
