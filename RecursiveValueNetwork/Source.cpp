#include <iostream>

void cpuSgemmStridedBatched(
	bool transB, bool transA,
	int CCols, int CRows, int AColsBRows,
	const float* alpha,
	float* B, int ColsB, int SizeB,
	float* A, int ColsA, int SizeA,
	const float* beta,
	float* C, int ColsC, int SizeC,
	int batchCount)
{
	for (int b = batchCount; b--;)
	{
		for (int m = CCols; m--;)
			for (int n = CRows; n--;)
			{
				float sum = 0;
				for (int k = AColsBRows; k--;)
					sum += (transA ? A[k * ColsA + n] : A[n * ColsA + k]) * (transB ? B[m * ColsB + k] : B[k * ColsB + m]);
				C[n * ColsC + m] = *alpha * sum + *beta * C[n * ColsC + m];
			}
		A += SizeA;
		B += SizeB;
		C += SizeC;
	}
}

void cpuRelu(float* data, int size)
{
	for (int i = 0; i < size; i++)
		data[i] = data[i] > 0 ? data[i] : 0;
}

void cpuReluGradient(float* gradient, float* data, int size)
{
	for (int i = 0; i < size; i++)
		gradient[i] = data[i] > 0 ? gradient[i] : 0;
}

void cpuSaxpy(int n, float a, float* x, float* y)
{
	for (int i = 0; i < n; i++)
		y[i] = a * x[i] + y[i];
}

void PrintMatrixf32(float* arr, uint32_t rows, uint32_t cols, const char* label)
{
	printf("%s:\n", label);
	for (uint32_t i = 0; i < rows; i++)
	{
		for (uint32_t j = 0; j < cols; j++)
			printf("%8.3f ", arr[i * cols + j]);
		printf("\n");
	}
	printf("\n");
}

int main()
{
	srand(time(nullptr));

	const float alpha = 1;
	const float beta = 0;

	const float spawnRange = 10;
	const float halfSpawnRange = spawnRange * 0.5f;
	const float spawnRangeScalar = spawnRange / RAND_MAX;

	const float moveRange = 1;
	const float halfMoveRange = moveRange * 0.5f;
	const float moveRangeScalar = moveRange / RAND_MAX;

	const int maxEpisodes = 1;
	const int maxSteps = 10;

	const int inputSize = 2;
	const int hiddenMemSize = 32;
	const int hiddenLayer1Size = 32;
	const int hiddenLayer2Size = 32;
	const int outputSize = 1;

	float x, y;
	float sqrDistances[maxSteps];
	float hiddenMemParam[hiddenMemSize];
	float hiddenMem[hiddenMemSize * maxSteps];
	float inputs[inputSize * maxSteps];
	float hiddenLayer1[hiddenLayer1Size * maxSteps];
	float hiddenLayer2[hiddenLayer1Size * maxSteps];
	float outputs[outputSize * maxSteps];

	float hiddenLayer1Weight[hiddenLayer1Size * inputSize];
	float hiddenLayer2Weight[hiddenLayer2Size * hiddenLayer1Size];
	float outputLayerWeight[outputSize * hiddenLayer2Size];

	// weight initialization
	for (int i = 0; i < hiddenLayer1Size * inputSize; ++i)
		hiddenLayer1Weight[i] = (float)rand() / RAND_MAX / inputSize;
	for (int i = 0; i < hiddenLayer2Size * hiddenLayer1Size; ++i)
		hiddenLayer2Weight[i] = (float)rand() / RAND_MAX / hiddenLayer1Size;
	for (int i = 0; i < outputSize * hiddenLayer2Size; ++i)
		outputLayerWeight[i] = (float)rand() / RAND_MAX / hiddenLayer2Size;

	//PrintMatrixf32(hiddenLayer1Weight, hiddenLayer1Size, inputSize, "hiddenLayer1Weight");

	for (int episode = 0; episode < maxEpisodes; ++episode)
	{
		x = rand() * spawnRangeScalar - halfSpawnRange;
		y = rand() * spawnRangeScalar - halfSpawnRange;
		for (int step = 0;step < maxSteps; ++step)
		{
			inputs[step * inputSize] = x;
			inputs[step * inputSize + 1] = y;
			sqrDistances[step] = x * x + y * y;

			//PrintMatrixf32(inputs + step * inputSize, 1, inputSize, "inputs");

			cpuSgemmStridedBatched
			(
				false, false,
				hiddenLayer1Size, 1, inputSize,
				&alpha,
				hiddenLayer1Weight, hiddenLayer1Size, 0,
				inputs + step * inputSize, inputSize, 0,
				&beta,
				hiddenLayer1 + step * hiddenLayer1Size, hiddenLayer1Size, 0,
				1
			);

			//PrintMatrixf32(hiddenLayer1 + step * hiddenLayer1Size, 1, hiddenLayer1Size, "hiddenLayer1");

			cpuRelu(hiddenLayer1 + step * hiddenLayer1Size, hiddenLayer1Size);

			cpuSgemmStridedBatched
			(
				false, false,
				hiddenLayer2Size, 1, hiddenLayer1Size,
				&alpha,
				hiddenLayer2Weight, hiddenLayer2Size, 0,
				hiddenLayer1 + step * hiddenLayer1Size, hiddenLayer1Size, 0,
				&beta,
				hiddenLayer2 + step * hiddenLayer2Size, hiddenLayer2Size, 0,
				1
			);

			//PrintMatrixf32(hiddenLayer2 + step * hiddenLayer2Size, 1, hiddenLayer2Size, "hiddenLayer2");

			cpuRelu(hiddenLayer2 + step * hiddenLayer2Size, hiddenLayer2Size);

			cpuSgemmStridedBatched
			(
				false, false,
				outputSize, 1, hiddenLayer2Size,
				&alpha,
				outputLayerWeight, outputSize, 0,
				hiddenLayer2 + step * hiddenLayer2Size, hiddenLayer2Size, 0,
				&beta,
				outputs + step * outputSize, outputSize, 0,
				1
			);

			//PrintMatrixf32(outputs + step * outputSize, 1, outputSize, "outputs");

			printf("x: %f, y: %f, sqrDist: %f, output: %f\n", x, y, sqrDistances[step], outputs[step * outputSize]);

			x += rand() * moveRangeScalar - halfMoveRange;
			y += rand() * moveRangeScalar - halfMoveRange;
		}
	}

	return 0;
}

/*
point = np.array([[3, 4]])  # note the double square brackets; this creates a 2D array
squared_distance = model.predict(point)

print(f"Point: {point}")
print(f"Squared distance: {squared_distance}")
print(f"Expected squared distance: {np.sum(point ** 2)}")
*/