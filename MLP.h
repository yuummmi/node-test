#pragma once

#define LEARNING_RATE	0.1

class CMLP
{
public:
	CMLP();
	~CMLP();

	
	int m_iNumInNodes;
	int m_iNumOutNodes;
	int m_iNumHiddenLayer;	
	int m_iNumTotalLayer;	
	int* m_NumNodes;		

	double*** m_Weight;
	double** m_NodeOut;				

	double** m_ErrorGradient;
	double* pInValue, * pOutValue;
	double* pCorrectOutValue;		

	bool Create(int InNode, int* pHiddenNode, int OutNode, int HiddenLayer);
private:
	void InitW();
	double ActivationFunc(double u);

public:
	void Forward();

	void BackPropagationLearning();
	bool SaveWeight(char* fname);
	bool LoadWeight(char* fname);
};


