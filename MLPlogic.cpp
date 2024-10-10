#include <stdio.h>

#include "MLP.h"
CMLP MultiLayer;

#define NUM_TRAINING_SET	8
#define NUM_INPUT		3
#define NUM_OUTPUT		2
#define MAX_EPOCH	1000000

int main()
{
	int n;

	/ 다층신경망 구조 선언
		int HiddenNodes[2] = { 3, 2 };
	int numofHiddenLayer = 2;
	MultiLayer.Create(NUM_INPUT, HiddenNodes, NUM_OUTPUT, numofHiddenLayer);
	double x[NUM_TRAINING_SET][NUM_INPUT] = { { 0,0,0},{ 0,0,1 },{ 0,1,0 },{ 0,1,1 },
		{ 1,0,0 },{ 1,0,1 },{ 1,1,0 },{ 1,1,1 } };	//입력
	double d[NUM_TRAINING_SET][NUM_OUTPUT] = { {0,1},{0,0},{1,0},{0,0},{1,0},{0,1},{1,1},{0,0} }; //정답

	if (MultiLayer.LoadWeight("weight.txt"))
		printf("기존의 가중치로부터 학습을 시작합니다.\n");
	else
		printf("램덤 가중치로부터 처음으로 시작합니다.\n");

	// 학습전 결과출력
	for (n = 0; n < NUM_TRAINING_SET; n++) {
		for (int i = 1; i <= NUM_INPUT; i++)	MultiLayer.pInValue[i] = x[n][i - 1];
		//MultiLayer.pInValue[1] = x[n][0];
		//MultiLayer.pInValue[2] = x[n][1];
		//MultiLayer.pInValue[3] = x[n][2];

		MultiLayer.Forward();

		for (int i = 1; i <= NUM_INPUT; i++)	printf("%lf ", MultiLayer.pInValue[i]);
		printf("= ");
		for (int out_no = 1; out_no <= NUM_OUTPUT; out_no++) printf("%lf ", MultiLayer.pOutValue[out_no]);
		printf("\n");
	}
	getchar();

	// 학습
	printf("\n=>학습을 시작합니다.**********************\n");
	int i, j, p;
	int epoch;
	double MSE;
	for (epoch = 0; epoch < MAX_EPOCH; epoch++)
	{
		MSE = 0.0;
		// 입력과 정답을 전달
		for (i = 0; i < NUM_TRAINING_SET; i++)
		{
			//입력전달
			for (j = 0; j < NUM_INPUT; j++)
				MultiLayer.pInValue[j + 1] = x[i][j];
			//정답전달
			MultiLayer.pCorrectOutValue[1] = dout[i][0];

			// 출력값계산
			MultiLayer.Forward();
			// 역전파학습
			MultiLayer.BackPropagationLearning();

			// 갱신후에 출력과 에러값을 계산
			MultiLayer.Forward();
			for (p = 1; p <= NUM_OUTPUT; p++)
				MSE += (MultiLayer.pCorrectOutValue[p] - MultiLayer.pOutValue[p]) * (MultiLayer.pCorrectOutValue[p] - MultiLayer.pOutValue[p]);

		}
		MSE /= NUM_TRAINING_SET;
		printf("Epoch%d(MSE)=%.15f\n", epoch, MSE);
	}
	printf("\n=>학습이 완료되었습니다.************************\n");


	MultiLayer.SaveWeight("weight.txt");

	// 학습후 결과출력
	printf("학습후 결과\n");
	for (n = 0; n < NUM_TRAINING_SET; n++)
	{
		for (int i = 1; i <= NUM_INPUT; i++)				MultiLayer.pInValue[i] = x[n][i - 1];

		MultiLayer.Forward();

		for (int i = 1; i <= NUM_INPUT; i++)
			printf("%lf ", MultiLayer.pInValue[i]);
		printf("= ");
		for (int out_no = 1; out_no <= NUM_OUTPUT; out_no++) 			printf("%lf ", MultiLayer.pOutValue[out_no]);
		printf("\n");
	}
	return 0;
}










