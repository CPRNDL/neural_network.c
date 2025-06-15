#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define TRAIN_DATA 10000
#define TEST_DATA 1000
#define INPUT 784
#define HIDDEN 150
#define OUTPUT 4
#define ETA 0.007
#define EPOCH 10

double x_train[TRAIN_DATA][INPUT];
int y_train[TRAIN_DATA];
double x_test[TEST_DATA][INPUT];
int y_test[TEST_DATA];

double w_ih[HIDDEN][INPUT], w_ho[OUTPUT][HIDDEN];
double b_h[HIDDEN], b_o[OUTPUT];
double a_h[HIDDEN], a_o[OUTPUT];
double d_h[HIDDEN], d_o[OUTPUT];

double sigmoid(double z) {
    return 1.0 / (1.0 + exp(-z));
}

double random_weight(int n) {
    double limit = sqrt(1.0 / n);
    return ((double)rand() / RAND_MAX) * 2 * limit - limit;
}

void load_data(const char* filename, double x[][INPUT], int y[], int size) {
    FILE* fp = fopen(filename, "r");
    if (!fp) { puts("File open error"); exit(1); }

    for (int i = 0; i < size; i++) {
        fscanf(fp, "%d", &y[i]);
        for (int j = 0; j < INPUT; j++) {
            int temp;
            fscanf(fp, "%d", &temp);
            x[i][j] = temp / 255.0;
        }
    }
    fclose(fp);
}

void init_parameters() {
    for (int i = 0; i < HIDDEN; i++) {
        b_h[i] = random_weight(INPUT);
        for (int j = 0; j < INPUT; j++)
            w_ih[i][j] = random_weight(INPUT);
    }
    for (int i = 0; i < OUTPUT; i++) {
        b_o[i] = random_weight(HIDDEN);
        for (int j = 0; j < HIDDEN; j++)
            w_ho[i][j] = random_weight(HIDDEN);
    }
}

void forward_train(int index) {
    for (int i = 0; i < HIDDEN; i++) {
        double sum = b_h[i];
        for (int j = 0; j < INPUT; j++)
            sum += w_ih[i][j] * x_train[index][j];
        a_h[i] = sigmoid(sum);
    }

    for (int i = 0; i < OUTPUT; i++) {
        double sum = b_o[i];
        for (int j = 0; j < HIDDEN; j++)
            sum += w_ho[i][j] * a_h[j];
        a_o[i] = sigmoid(sum);
    }
}

void forward_test(int index) {
    for (int i = 0; i < HIDDEN; i++) {
        double sum = b_h[i];
        for (int j = 0; j < INPUT; j++)
            sum += w_ih[i][j] * x_test[index][j];
        a_h[i] = sigmoid(sum);
    }

    for (int i = 0; i < OUTPUT; i++) {
        double sum = b_o[i];
        for (int j = 0; j < HIDDEN; j++)
            sum += w_ho[i][j] * a_h[j];
        a_o[i] = sigmoid(sum);
    }
}

double backward(int label, int index) {
    double cost = 0.0;
    for (int i = 0; i < OUTPUT; i++) {
        int bit = (label >> i) & 1;
        double t = (double)bit;
        cost += 0.5 * pow(t - a_o[i], 2);
        d_o[i] = (a_o[i] - t) * a_o[i] * (1 - a_o[i]);
    }

    for (int i = 0; i < HIDDEN; i++) {
        double sum = 0.0;
        for (int j = 0; j < OUTPUT; j++)
            sum += d_o[j] * w_ho[j][i];
        d_h[i] = sum * a_h[i] * (1 - a_h[i]);
    }

    for (int i = 0; i < OUTPUT; i++) {
        for (int j = 0; j < HIDDEN; j++)
            w_ho[i][j] -= ETA * d_o[i] * a_h[j];
        b_o[i] -= ETA * d_o[i];
    }

    for (int i = 0; i < HIDDEN; i++) {
        for (int j = 0; j < INPUT; j++)
            w_ih[i][j] -= ETA * d_h[i] * x_train[index][j];
        b_h[i] -= ETA * d_h[i];
    }

    return cost;
}

int decode_output(double output[OUTPUT]){
    int val = 0;
    for(int i=0; i<OUTPUT; i++){
        int bit = (output[i] >= 0.5) ? 1 : 0;  // threshold 0.5로 통일
        val += bit << i;
    }
    return val;
}

int predict(double x[INPUT]) {
    for (int i = 0; i < HIDDEN; i++) {
        double sum = b_h[i];
        for (int j = 0; j < INPUT; j++)
            sum += w_ih[i][j] * x[j];
        a_h[i] = sigmoid(sum);
    }

    for (int i = 0; i < OUTPUT; i++) {
        double sum = b_o[i];
        for (int j = 0; j < HIDDEN; j++)
            sum += w_ho[i][j] * a_h[j];
        a_o[i] = sigmoid(sum);
    }

    return decode_output(a_o);
}

void shuffle_data() {
    for (int i = TRAIN_DATA - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        for (int k = 0; k < INPUT; k++) {
            double tmp = x_train[i][k];
            x_train[i][k] = x_train[j][k];
            x_train[j][k] = tmp;
        }
        int tmp = y_train[i];
        y_train[i] = y_train[j];
        y_train[j] = tmp;
    }
}

int main() {
    srand((unsigned)time(NULL));
    load_data("train.txt", x_train, y_train, TRAIN_DATA);
    load_data("test.txt", x_test, y_test, TEST_DATA);
    init_parameters();

    for (int e = 0; e < EPOCH; e++) {
        shuffle_data();
        double cost = 0.0;
        for (int i = 0; i < TRAIN_DATA; i++) {
            forward_train(i);
            cost += backward(y_train[i], i);
        }
        printf("[Epoch %d] Cost: %.6lf\n", e + 1, cost / TRAIN_DATA);
    }

    int correct = 0;
    for (int i = 0; i < TEST_DATA; i++) {
        forward_test(i);
        int pred = decode_output(a_o);
        int label = y_test[i];
        if (pred == label) correct++;
    }
    printf("Accuracy: %.2lf%%\n", 100.0 * correct / TEST_DATA);

    return 0;
}
