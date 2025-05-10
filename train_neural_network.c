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
#define DATA 1000      //데이터 수
#define INPUT 784       //데이터 크기 -> 784 = 28 * 28
#define HIDDEN 20       //은닉층 노드 수
#define OUTPUT 4        //출력층 노드 수 -> 출력층은 0~1 사이의 값을 가지기 때문에 10진수 숫자 예측을 위해 4비트로 설정
#define ETA 0.1         //학습률(η)
#define EPOCH 100       //학습 횟수


//전역 변수 선언
double train[DATA][INPUT];      //학습 데이터
int label[DATA];                //정답 데이터


//함수 파라미터를 관리하기 쉽게 구조체 정의
typedef struct {
    double weight1[HIDDEN][INPUT];  //입력층 -> 은닉층 가중치
    double weight2[OUTPUT][HIDDEN]; //은닉층 -> 출력층 가중치
    double bias1[HIDDEN];           //은닉층 편향
    double bias2[OUTPUT];           //출력층 편향
    double hiddenS[HIDDEN];         //은닉층의 시그모이드 값
    double outputS[OUTPUT];         //출력층의 시그모이드 값
    double hidden_delta[HIDDEN];    //은닉층의 오차
    double output_delta[OUTPUT];    //출력층의 오차
} Parameters;


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


//데이터 파일 읽기 함수
void LoadInputData(const char* filename){           //파일 이름이 배열의 포인터를 가리키므로 const char 포인터로 받음
    FILE *fp = fopen(filename, "r");                //fopen으로 파일 열기 -> return 주소 값을 FILE 구조체 포인터에 넣음
    if(fp==NULL){                                   //포인터가 NULL을 반환 -> 인식 못 함
        printf("Failed to open file\n");            //파일 열기 실패 출력
        exit(1);                                    //비정상 종료
    }
    for(int i=0; i<DATA; i++){                      //데이터 수 만큼 반복
        int temp;                                   //임시 변수
        if(fscanf(fp, "%d", &temp)!=1){             //fscanf 실패시
            printf("Error in reading file\n");      //오류 메시지 출력
            exit(1);                                //비정상 종료
        }
        label[i]=temp;                              //정답 데이터 넣기
        for(int j=0; j<INPUT; j++){                 //28*28 만큼 반복
            if(fscanf(fp, "%d", &temp)!=1){         //fscanf 실패시
                printf("Error in reading file\n");  //오류 메시지 출력
                exit(1);                            //비정상 종료
            }
            train[i][j]=(double)temp/255;
        }
    }
    fclose(fp);     //파일 닫기
}


//초기 가중치 편향 설정 함수
void RandomizeParameters(Parameters* p){
    for(int i=0; i<HIDDEN; i++){
        p->bias1[i]=RandomGeneration(INPUT);
        for(int j=0; j<INPUT; j++){
            p->weight1[i][j]=RandomGeneration(INPUT);
        }
    }
    for(int i=0; i<OUTPUT; i++){
        p->bias2[i]=RandomGeneration(HIDDEN);
        for(int j=0; j<HIDDEN; j++){
            p->weight2[i][j]=RandomGeneration(HIDDEN);
        }
    }
}


//데이터 학습 루프 함수들
//순전파
void Forward(Parameters* p, int index){
    double sum;
    for(int i=0; i<HIDDEN; i++){
        sum=p->bias1[i];
        for(int j=0; j<INPUT; j++){
            sum+=p->weight1[i][j]*train[index][j];
        }
        p->hiddenS[i]=Sigmoid(sum);
    }

    for(int i=0; i<OUTPUT; i++){
        sum=p->bias2[i];
        for(int j=0; j<HIDDEN; j++){
            sum+=p->weight2[i][j]*p->hiddenS[j];
        }
        p->outputS[i]=Sigmoid(sum);
    }
}

//오차역전파
double Backward(Parameters* p, int label){
    double C=0.0;
    for(int i=0; i<OUTPUT; i++){
        int temp1=(label>>i)&1;
        double temp2=(double)temp1;
        double temp3=p->outputS[i];
        C+=0.5*pow(temp2-temp3, 2);
        p->output_delta[i]=(temp3-temp2)*temp3*(1-temp3);
    }
    for(int i=0; i<HIDDEN; i++){
        double sum=0.0;
        for(int j=0; j<OUTPUT; j++){
            sum+=p->output_delta[j]*p->weight2[j][i];
        }
        double temp=p->hiddenS[i];
        p->hidden_delta[i]=sum*temp*(1-temp);
    }
    return C;
}

//파라미터 업데이트 함수
void UpdateParameters(Parameters* p, int index){
    for(int i=0; i<OUTPUT; i++){
        for(int j=0; j<HIDDEN; j++){
            p->weight2[i][j]-=ETA*p->output_delta[i]*p->hiddenS[j];
        }
        p->bias2[i]-=ETA*p->output_delta[i];
    }
    for(int i=0; i<HIDDEN; i++){
        for(int j=0; j<INPUT; j++){
            p->weight1[i][j]-=ETA*p->hidden_delta[i]*train[index][j];
        }
        p->bias1[i]-=ETA*p->hidden_delta[i];
    }
}


int main(){
    LoadInputData("train.txt");         //데이터 읽기
    srand((unsigned int)time(NULL));    //현재 시간을 시드로 사용
    
    Parameters p;
    RandomizeParameters(&p);

    for(int epoch=0; epoch<EPOCH; epoch++){
        double epoch_C=0.0;
        for(int i=0; i<DATA; i++){
            int label_index=label[i];
            Forward(&p, i);
            epoch_C+=Backward(&p, label_index);
            UpdateParameters(&p, i);
        }
        printf("[Epoch %d] Cost: %lf\n", epoch+1, epoch_C/DATA);
    }
}