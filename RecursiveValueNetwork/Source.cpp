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
	const float learningRate = 1.0f;

	const float beta1 = 0.9f;
	const float beta2 = 0.999f;
	const float epsilon = 1e-8f;

	const float spawnRange = 10;
	const float halfSpawnRange = spawnRange * 0.5f;
	const float spawnRangeScalar = spawnRange / RAND_MAX;

	const float moveRange = 1;
	const float halfMoveRange = moveRange * 0.5f;
	const float moveRangeScalar = moveRange / RAND_MAX;

	const int maxEpisodes = 1000;
	const int maxSteps = 10;
	const int batchSize = 32;

	const int hiddenMemSize = 16;
	const int inputSize = 2;
	const int hiddenLayer1Size = 64;
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

	float inputGradients[inputSize * maxSteps];
	float hiddenLayer1Gradients[hiddenLayer1Size * maxSteps];
	float hiddenLayer2Gradients[hiddenLayer1Size * maxSteps];
	float outputGradients[outputSize * maxSteps];

	float hiddenLayer1WeightGradients[hiddenLayer1Size * inputSize];
	float hiddenLayer2WeightGradients[hiddenLayer2Size * hiddenLayer1Size];
	float outputLayerWeightGradients[outputSize * hiddenLayer2Size];

	float exponentiallyDecayedMean = 1;
	float exponentiallyDecayedVariance = 1;

	float hiddenLayer1WeightGradientMean[hiddenLayer1Size * inputSize];
	float hiddenLayer2WeightGradientMean[hiddenLayer2Size * hiddenLayer1Size];
	float outputLayerWeightGradientMean[outputSize * hiddenLayer2Size];

	float hiddenLayer1WeightGradientVariance[hiddenLayer1Size * inputSize];
	float hiddenLayer2WeightGradientVariance[hiddenLayer2Size * hiddenLayer1Size];
	float outputLayerWeightGradientVariance[outputSize * hiddenLayer2Size];

	// set gradients mean/varience to 0
	memset(hiddenLayer1WeightGradientMean, 0, hiddenLayer1Size * inputSize * sizeof(float));
	memset(hiddenLayer2WeightGradientMean, 0, hiddenLayer2Size * hiddenLayer1Size * sizeof(float));
	memset(outputLayerWeightGradientMean, 0, outputSize * hiddenLayer2Size * sizeof(float));

	memset(hiddenLayer1WeightGradientVariance, 0, hiddenLayer1Size * inputSize * sizeof(float));
	memset(hiddenLayer2WeightGradientVariance, 0, hiddenLayer2Size * hiddenLayer1Size * sizeof(float));
	memset(outputLayerWeightGradientVariance, 0, outputSize * hiddenLayer2Size * sizeof(float));

	// weight initialization
	for (int i = 0; i < hiddenLayer1Size * inputSize; ++i)
		hiddenLayer1Weight[i] = ((float)rand() / RAND_MAX - 0.5) / inputSize;
	for (int i = 0; i < hiddenLayer2Size * hiddenLayer1Size; ++i)
		hiddenLayer2Weight[i] = ((float)rand() / RAND_MAX - 0.5) / hiddenLayer1Size;
	for (int i = 0; i < outputSize * hiddenLayer2Size; ++i)
		outputLayerWeight[i] = ((float)rand() / RAND_MAX - 0.5) / hiddenLayer2Size;

	//PrintMatrixf32(hiddenLayer1Weight, hiddenLayer1Size, inputSize, "hiddenLayer1Weight");

	for (int episode = 0; episode < maxEpisodes; ++episode)
	{
		float averageError = 0;

		exponentiallyDecayedMean *= beta1;
		exponentiallyDecayedVariance *= beta2;

		// reset all weight gradients
		memset(hiddenLayer1WeightGradients, 0, hiddenLayer1Size * inputSize * sizeof(float));
		memset(hiddenLayer2WeightGradients, 0, hiddenLayer2Size * hiddenLayer1Size * sizeof(float));
		memset(outputLayerWeightGradients, 0, outputSize * hiddenLayer2Size * sizeof(float));

		for (int batch = 0; batch < batchSize; ++batch)
		{
			x = rand() * spawnRangeScalar - halfSpawnRange;
			y = rand() * spawnRangeScalar - halfSpawnRange;

			for (int step = 0; step < maxSteps; ++step)
			{
				inputs[step * inputSize] = x;
				inputs[step * inputSize + 1] = y;
				sqrDistances[step] = x * x + y * y;
				//sqrDistances[step] = x + y;

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

				//printf("x: %f, y: %f, sqrDist: %f, output: %f\n", inputs[step * inputSize], inputs[step * inputSize + 1], sqrDistances[step], outputs[step * outputSize]);

				x += rand() * moveRangeScalar - halfMoveRange;
				y += rand() * moveRangeScalar - halfMoveRange;
			}

			for (int step = maxSteps; step--;)
			{
				//printf("x: %f, y: %f, sqrDist: %f, output: %f\n", inputs[step * inputSize], inputs[step * inputSize + 1], sqrDistances[step], outputs[step * outputSize]);

				float error = sqrDistances[step] - outputs[step * outputSize];
				averageError += abs(error);

				outputGradients[step * outputSize] = error;

				cpuSgemmStridedBatched
				(
					true, false,
					hiddenLayer2Size, 1, outputSize,
					&alpha,
					outputLayerWeight, outputSize, 0,
					outputGradients + step * outputSize, outputSize, 0,
					&beta,
					hiddenLayer2Gradients + step * hiddenLayer2Size, hiddenLayer2Size, 0,
					1
				);

				cpuSgemmStridedBatched
				(
					false, true,
					outputSize, hiddenLayer2Size, 1,
					&alpha,
					outputGradients + step * outputSize, outputSize, 0,
					hiddenLayer2 + step * hiddenLayer2Size, hiddenLayer2Size, 0,
					&alpha,
					outputLayerWeightGradients, outputSize, 0,
					1
				);

				cpuReluGradient(hiddenLayer2Gradients + step * hiddenLayer2Size, hiddenLayer2 + step * hiddenLayer2Size, hiddenLayer2Size);

				//PrintMatrixf32(hiddenLayer2Gradients + step * hiddenLayer2Size, 1, hiddenLayer2Size, "hiddenLayer2Gradients");
				//PrintMatrixf32(outputLayerWeightGradients, outputSize, hiddenLayer2Size, "outputLayerWeightGradients");

				cpuSgemmStridedBatched
				(
					true, false,
					hiddenLayer1Size, 1, hiddenLayer2Size,
					&alpha,
					hiddenLayer2Weight, hiddenLayer2Size, 0,
					hiddenLayer2Gradients + step * hiddenLayer2Size, hiddenLayer2Size, 0,
					&beta,
					hiddenLayer1Gradients + step * hiddenLayer1Size, hiddenLayer1Size, 0,
					1
				);

				cpuSgemmStridedBatched
				(
					false, true,
					hiddenLayer2Size, hiddenLayer1Size, 1,
					&alpha,
					hiddenLayer2Gradients + step * hiddenLayer2Size, hiddenLayer2Size, 0,
					hiddenLayer1 + step * hiddenLayer1Size, hiddenLayer1Size, 0,
					&alpha,
					hiddenLayer2WeightGradients, hiddenLayer2Size, 0,
					1
				);

				cpuReluGradient(hiddenLayer1Gradients + step * hiddenLayer1Size, hiddenLayer1 + step * hiddenLayer1Size, hiddenLayer1Size);

				//PrintMatrixf32(hiddenLayer1Gradients + step * hiddenLayer1Size, 1, hiddenLayer1Size, "hiddenLayer1Gradients");
				//PrintMatrixf32(hiddenLayer2WeightGradients, hiddenLayer2Size, hiddenLayer1Size, "hiddenLayer2WeightGradients");

				/*cpuSgemmStridedBatched
				(
					true, false,
					inputSize, 1, hiddenLayer1Size,
					&alpha,
					hiddenLayer1Weight, hiddenLayer1Size, 0,
					hiddenLayer1Gradients + step * hiddenLayer1Size, hiddenLayer1Size, 0,
					&beta,
					inputGradients + step * inputSize, inputSize, 0,
					1
				);*/

				cpuSgemmStridedBatched
				(
					false, true,
					hiddenLayer1Size, inputSize, 1,
					&alpha,
					hiddenLayer1Gradients + step * hiddenLayer1Size, hiddenLayer1Size, 0,
					inputs + step * inputSize, inputSize, 0,
					&alpha,
					hiddenLayer1WeightGradients, hiddenLayer1Size, 0,
					1
				);

				//PrintMatrixf32(inputGradients + step * inputSize, 1, inputSize, "inputGradients");
				//PrintMatrixf32(hiddenLayer1WeightGradients, hiddenLayer1Size, inputSize, "hiddenLayer1WeightGradients");
			}
		}

		printf("averageError: %f\n", averageError / (maxSteps * batchSize));

		// for every weight, calculate the mean/varience gradient
		for (int i = 0; i < outputSize * hiddenLayer2Size; i++)
		{
			float gradient = outputLayerWeightGradients[i] / (maxSteps * batchSize);
			outputLayerWeightGradientMean[i] = beta1 * outputLayerWeightGradientMean[i] + (1 - beta1) * gradient;
			outputLayerWeightGradientVariance[i] = beta2 * outputLayerWeightGradientVariance[i] + (1 - beta2) * gradient * gradient;

			float correctedMean = outputLayerWeightGradientMean[i] / (1 - exponentiallyDecayedMean);
			float correctedVariance = outputLayerWeightGradientVariance[i] / (1 - exponentiallyDecayedVariance);

			outputLayerWeight[i] += learningRate * correctedMean / (sqrt(correctedVariance) + epsilon);
		}
	}

	return 0;
}