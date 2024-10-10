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

	/ �����Ű�� ���� ����
		int HiddenNodes[2] = { 3, 2 };
	int numofHiddenLayer = 2;
	MultiLayer.Create(NUM_INPUT, HiddenNodes, NUM_OUTPUT, numofHiddenLayer);
	double x[NUM_TRAINING_SET][NUM_INPUT] = { { 0,0,0},{ 0,0,1 },{ 0,1,0 },{ 0,1,1 },
		{ 1,0,0 },{ 1,0,1 },{ 1,1,0 },{ 1,1,1 } };	//�Է�
	double d[NUM_TRAINING_SET][NUM_OUTPUT] = { {0,1},{0,0},{1,0},{0,0},{1,0},{0,1},{1,1},{0,0} }; //����

	if (MultiLayer.LoadWeight("weight.txt"))
		printf("������ ����ġ�κ��� �н��� �����մϴ�.\n");
	else
		printf("���� ����ġ�κ��� ó������ �����մϴ�.\n");

	// �н��� ������
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

	// �н�
	printf("\n=>�н��� �����մϴ�.**********************\n");
	int i, j, p;
	int epoch;
	double MSE;
	for (epoch = 0; epoch < MAX_EPOCH; epoch++)
	{
		MSE = 0.0;
		// �Է°� ������ ����
		for (i = 0; i < NUM_TRAINING_SET; i++)
		{
			//�Է�����
			for (j = 0; j < NUM_INPUT; j++)
				MultiLayer.pInValue[j + 1] = x[i][j];
			//��������
			MultiLayer.pCorrectOutValue[1] = dout[i][0];

			// ��°����
			MultiLayer.Forward();
			// �������н�
			MultiLayer.BackPropagationLearning();

			// �����Ŀ� ��°� �������� ���
			MultiLayer.Forward();
			for (p = 1; p <= NUM_OUTPUT; p++)
				MSE += (MultiLayer.pCorrectOutValue[p] - MultiLayer.pOutValue[p]) * (MultiLayer.pCorrectOutValue[p] - MultiLayer.pOutValue[p]);

		}
		MSE /= NUM_TRAINING_SET;
		printf("Epoch%d(MSE)=%.15f\n", epoch, MSE);
	}
	printf("\n=>�н��� �Ϸ�Ǿ����ϴ�.************************\n");


	MultiLayer.SaveWeight("weight.txt");

	// �н��� ������
	printf("�н��� ���\n");
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










