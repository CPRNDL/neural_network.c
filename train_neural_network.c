#include <stdio.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include <stdlib.h>
#include <time.h>

#define DATA 600000
#define INPUT 784
#define HIDDEN 20
#define OUTPUT 4
#define ETA 0.1
#define EPOCH 100

double train[DATA][INPUT];

//활성화 함수: 시그모이드 함수
double Sigmoid(double z){
    return 1/(1+exp(-z));
}

//limit을 갖는 난수 생성 함수
double RandomGeneration(int n_in) {
    double limit = sqrt(1.0/n_in);
    return ((double)rand()/RAND_MAX)*2*limit-limit;
}

//데이터 파일 읽기
void LoadInputData(const char* filename){ //파일 이름이 배열의 포인터를 가리키므로 const char 포인터로 받음
    FILE *fp = fopen(filename, "r"); //fopen으로 파일 열기 -> return 주소 값을 FILE 구조체 포인터에 넣음
    if(fp==NULL){ //포인터가 NULL을 반환 -> 인식 못 함
        printf("Failed to open file\n"); //파일 열기 실패 출력
        exit(1); //비정상 종료
    }
    for(int i=0; i<DATA; i++){ //데이터 수 만큼 반복
        int temp;
        if(fscanf(fp, "%d", &temp)!=1) //fscanf 실패시
        for(int j=0; j<INPUT; j++){ //28*28 만큼 반복
            int temp;
            if(fscanf(fp, "%d", &temp)!=1){
                printf("Error in reading file\n");
                exit(1);
            }
            train[i][j]=(double)temp;
        }
    }
    fclose(fp);
}

int main(){
    srand((unsigned int)time(NULL)); //현재 시간을 시드로 사용

    double hidden[HIDDEN];
    double hiddenS[HIDDEN];
    double outputS[OUTPUT];
    double weight1[HIDDEN][INPUT];
    double weight2[OUTPUT][HIDDEN];
    double bias1[HIDDEN];
    double bias2[OUTPUT];
    double teach[10][OUTPUT]={
        {0,0,0,0},
        {1,0,0,0},
        {0,1,0,0},
        {1,1,0,0},
        {0,0,1,0},
        {1,0,1,0},
        {0,1,1,0},
        {1,1,1,0},
        {0,0,0,1},
        {1,0,0,1}
    };
    double hidden_delta[HIDDEN];
    double output_delta[OUTPUT];
    double sum=0, C=0, epoch_cost=0;
    int teach_indexing, data_amount=DATA/10;

    LoadInputData("digit_data.txt");
    
    for(int i=0; i<HIDDEN; i++){
        bias1[i]=RandomGeneration(INPUT);
        for(int j=0; j<INPUT; j++){
            weight1[i][j]=RandomGeneration(INPUT);
        }
    }
    for(int i=0; i<OUTPUT; i++){
        bias2[i]=RandomGeneration(INPUT);
        for(int j=0; j<HIDDEN; j++){
            weight2[i][j]=RandomGeneration(INPUT);
        }
    }

    for(int epoch=0; epoch<EPOCH; epoch++){
        epoch_cost=0;
        for(int data_size=0; data_size<DATA; data_size++){
            teach_indexing=data_size/data_amount;

            //입력층 -> 은닉층 계산
            for(int i=0; i<HIDDEN; i++){
                sum=0;
                for(int j=0; j<INPUT; j++){
                    sum+=weight1[i][j]*train[data_size][j];
                }
                sum+=bias1[i];
                hidden[i]=sum;
                hiddenS[i]=Sigmoid(sum);
            }
        
            //은닉층 -> 출력층 계산
            for(int i=0; i<OUTPUT; i++){
                sum=0;
                for(int j=0; j<HIDDEN; j++){
                    sum+=weight2[i][j]*hidden[j];
                }
                sum+=bias2[i];
                outputS[i]=Sigmoid(sum);
            }

            C=0;
            for(int i=0; i<OUTPUT; i++){
                C+=0.5*pow(teach[teach_indexing][i]-outputS[i], 2);
                output_delta[i]=(outputS[i]-teach[teach_indexing][i])*outputS[i]*(1-outputS[i]);
            }
        
            for(int i=0; i<HIDDEN; i++){
                sum=0;
                for(int j=0; j<OUTPUT; j++){
                    sum+=output_delta[j]*weight2[j][i];
                }
                hidden_delta[i]=sum*hiddenS[i]*(1-hiddenS[i]);
            }
        
            for(int j=0; j<OUTPUT; j++){
                for(int i=0; i<HIDDEN; i++){
                    weight2[j][i]-=ETA*output_delta[j]*hiddenS[i];
                }
                bias2[j]-=ETA*output_delta[j];
            }
        
            for(int j=0; j<HIDDEN; j++){
                for(int i=0; i<INPUT; i++){
                    weight1[j][i]-=ETA*hidden_delta[j]*train[data_size][i];
                }
                bias1[j]-=ETA*hidden_delta[j];
            }
            
            epoch_cost+=C;
        }
        printf("[Epoch %d] Cost: %lf\n", epoch+1, epoch_cost/DATA);
    }

    double test_data[INPUT];
    int n;
    double s=0.0;

    printf("Input Data:\n");
    for(int i=0; i<INPUT; i++){
        scanf("%d", &n);
        test_data[i]=(double)n;
    }

    for(int i=0; i<HIDDEN; i++){
        sum=0;
        for(int j=0; j<INPUT; j++){
            sum+=weight1[i][j]*test_data[j];
        }
        sum+=bias1[i];
        hidden[i]=sum;
        hiddenS[i]=Sigmoid(sum);
    }

    for(int i=0; i<OUTPUT; i++){
        sum=0;
        for(int j=0; j<HIDDEN; j++){
            sum+=weight2[i][j]*hiddenS[j];
        }
        sum+=bias2[i];
        outputS[i]=Sigmoid(sum);
    }

    for(int i=0; i<OUTPUT; i++){
        if(outputS[i]>=0.5){
            s+=pow(2, i);
        }
    }
    printf("\nAnswer: %d", n);
}