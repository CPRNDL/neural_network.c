/*
프로그래밍 C반 수행평가
주제: 28 * 28 필기체 숫자 인식 인공 신경망 구현
30509민재헌
*/

//라이브러리 include
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>

//상수 매크로 선언: 여러 상수들을 조정하기 쉽도록
#define DATA 600000     //데이터 수
#define INPUT 784       //데이터 크기 -> 784 = 28 * 28
#define HIDDEN 20       //은닉층 노드 수
#define OUTPUT 4        //출력층 노드 수 -> 출력층은 0~1 사이의 값을 가지기 때문에 10진수 숫자 예측을 위해 4비트로 설정
#define ETA 0.1         //학습률(η)
#define EPOCH 100       //학습 횟수

//전역 변수 선언
double train[DATA][INPUT];  //학습 데이터
double answer[DATA];        //정답 데이터(int로 선언해도 상관 없지만 출력층과의 숫자 비교를 쉽게 하기 위해 double로 선언)

//활성화 함수: 시그모이드 함수
double Sigmoid(double z){
    return 1/(1+exp(-z));   //시그모이드 함숫값을 반환
}

//초기 가중치, 편향 설정을 위한 난수 생성 함수: Xavier Initialization
double RandomGeneration(int n) {
    double limit = sqrt(1.0/n);                         //역전파 기울기 소실과 폭주를 막기 위해 노드 수가 많을수록 범위값이 줄어들도록 설정
    return ((double)rand()/RAND_MAX)*2*limit-limit;     //rand() 함수는 0~RAND_MAX를 반환하므로 이를 RAND_MAX로 나누면 0~1 사이의 수가 됨
                                                        //*2*limit-limit -> -limit ~ +limit 범위로 변환
}

//데이터 파일 읽기
void LoadInputData(const char* filename){           //파일 이름이 배열의 포인터를 가리키므로 const char 포인터로 받음
    FILE *fp = fopen(filename, "r");                //fopen으로 파일 열기 -> return 주소 값을 FILE 구조체 포인터에 넣음
    if(fp==NULL){                                   //포인터가 NULL을 반환 -> 인식 못 함
        printf("Failed to open file\n");            //파일 열기 실패 출력
        exit(1);                                    //비정상 종료
    }
    for(int i=0; i<DATA; i++){                      //데이터 수 만큼 반복
        int temp;                                   //임시 변수
        answer[i]=(double)temp;                     //정답 데이터 넣기
        if(fscanf(fp, "%d", &temp)!=1){             //fscanf 실패시
            printf("Error in reading file\n");      //오류 메시지 출력
            exit(1);                                //비정상 종료
        }
        for(int j=0; j<INPUT; j++){                 //28*28 만큼 반복
            if(fscanf(fp, "%d", &temp)!=1){         //fscanf 실패시
                printf("Error in reading file\n");  //오류 메시지 출력
                exit(1);                            //비정상 종료
            }
            train[i][j]=(double)temp;               //여러 연산을 위해 데이터를 double로 넣기
        }
    }
    fclose(fp);     //파일 닫기
}


/*
데이터 학습 루프 함수
*/


int main(){
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

    srand((unsigned int)time(NULL)); //현재 시간을 시드로 사용
    
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

    for(int epoch=0; epoch<EPOCH; epoch++){                     //학습 시작
        epoch_cost=0;
        for(int data_size=0; data_size<DATA; data_size++){      //순방향 연산
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