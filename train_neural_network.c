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


//상수 매크로 선언
#define DATA 1000       //데이터 수
#define INPUT 784       //데이터 크기 -> 784 = 28 * 28
#define HIDDEN 20       //은닉층 노드 수
#define OUTPUT 4        //출력층 노드 수 -> 출력층은 0~1 사이의 값을 가지기 때문에 10진수 숫자 예측을 위해 4비트로 설정
#define ETA 0.1         //학습률(η)
#define EPOCH 100       //학습 횟수


//전역 변수 선언
double x_train[DATA][INPUT];    //학습 데이터
int y_train[DATA];              //학습 정답 데이터
double x_test[DATA][INPUT];     //테스트 데이터
int y_test[DATA];               //테스트 정답 데이터

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
void load_data(const char* filename){           //파일 이름이 배열의 포인터를 가리키므로 const char 포인터로 받음
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
        y_train[i]=temp;                              //정답 데이터 넣기
        for(int j=0; j<INPUT; j++){                 //28*28 만큼 반복
            if(fscanf(fp, "%d", &temp)!=1){         //fscanf 실패시
                printf("Error in reading file\n");  //오류 메시지 출력
                exit(1);                            //비정상 종료
            }
            x_train[i][j]=(double)temp/255;         //데이터 넣기
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
void forward(int index){                        //k번째 데이터를 입력으로 받음
    double sum;                                 //누적합 선언
    for(int i=0; i<HIDDEN; i++){
        sum=b_h[i];                             //누적합에 은닉층 편향 더함
        for(int j=0; j<INPUT; j++){
            sum+=w_ih[i][j]*x_train[index][j];  //누적합에 은닉층 가중입력값을 더함
        }
        a_h[i]=sigmoid(sum);                    //은닉층에 누적합의 활성화함숫값을 대입
    }

    for(int i=0; i<OUTPUT; i++){
        sum=b_o[i];                             //누적합에 출력층 편향 더함
        for(int j=0; j<HIDDEN; j++){
            sum+=w_ho[i][j]*a_h[j];             //누적합에 출력층 가중입력값을 더함
        }
        a_o[i]=sigmoid(sum);                    //출력층에 누적합의 활성화함숫값을 대입
    }
}

//오차역전파
double backward(int label){                     //정답 레이블을 받음
    double C=0.0;                               //비용함숫값 선언
    for(int i=0; i<OUTPUT; i++){
        int temp1=(label>>i)&1;                 //오른쪽 비트시프트 값이 1이면 1, 0이면 0
        double temp2=(double)temp1;             //temp1을 double로 저장
        double temp3=a_o[i];                    //출력층값
        C+=0.5*pow(temp2-temp3, 2);             //비용함숫값에 레이블과 출력층값의 차를 제곱하고 1/2을 곱한 값을 더함
        d_o[i]=(temp3-temp2)*temp3*(1-temp3);   //출력층의 오차값(비용함수의 출력층 편미분값)
    }
    for(int i=0; i<HIDDEN; i++){
        double sum=0.0;                         //누적합 선언
        for(int j=0; j<OUTPUT; j++){
            sum+=d_o[j]*w_ho[j][i];             //누적합에 출력층 오차와 은닉층->출력층 가중치를 곱한 값을 더함
        }
        double temp=a_h[i];                     //출력층 가중입력값
        d_h[i]=sum*temp*(1-temp);               //은닉층의 오차값(비용함수의 가중치 편미분값, 비용함수의 편향 편미분값)
    }
    return C;                                   //최종 비용함숫값 return
}

//파라미터 업데이트 함수
void update_parameters(int index){
    for(int i=0; i<OUTPUT; i++){
        for(int j=0; j<HIDDEN; j++){
            w_ho[i][j]-=ETA*d_o[i]*a_h[j];
        }
        b_o[i]-=ETA*d_o[i];
    }
    for(int i=0; i<HIDDEN; i++){
        for(int j=0; j<INPUT; j++){
            w_ih[i][j]-=ETA*d_h[i]*x_train[index][j];
        }
        b_h[i]-=ETA*d_h[i];
    }
}


int main(){
    load_data("train.txt");         //데이터 읽기
    srand((unsigned int)time(NULL));    //현재 시간을 시드로 사용

    init_parameters();

    for(int epoch=0; epoch<EPOCH; epoch++){
        double epoch_C=0.0;
        for(int i=0; i<DATA; i++){
            int label_index=y_train[i];
            forward(i);
            epoch_C+=backward(label_index);
            update_parameters(i);
        }
        printf("[Epoch %d] Cost: %lf\n", epoch+1, epoch_C/DATA);
    }
}