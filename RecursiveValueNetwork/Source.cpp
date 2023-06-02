#include <iostream>

/*
Important Lessons:
- for deeper or more iterations in rnns, use larger batchsizes and smaller learning rates
- not to sure if batchsize works because average usually decreases the variance of the gradients, sort of acting like smaller learning rates
- more rnn iterations is basically like training a network with more layers
- parameter initialization is very important, it helps speed up training and helps avoid local minima
- lower learning rate may speed up training a lot. There is a sort of sweet spot for learning rate, too high and it will diverge, too low and it will take forever to train
- adam doesn't desend as extremely as gradient descent, may be better for certain problems
- pros of adam: less parameter tuning required due to adaptive gradient step for each parameter
- pros of gradient descent: if you know the optimal learning rate, it will converge faster than adam
- adam deals with first and second moment of the gradient.
(instead of variance, it is more like an uncentered variance, allowing small gradients to have a larger effect and vice versa)
(different to centered variance, which is the average of the squared differences from the mean)
*/

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

float InvSqrt(float number)
{
	long i = 0x5F1FFFF9 - (*(long*)&number >> 1);
	float tmp = *(float*)&i;
	return tmp;
	return tmp * 0.703952253f * (2.38924456f - number * tmp * tmp);
}

void orthoganalInitialization(float* arr, int rows, int cols)
{
}

int main()
{
	srand(time(nullptr));

	const float alpha = 1;
	const float beta = 0;
	const float learningRate = 0.00001f;

	const float beta1 = 0.9f;
	const float beta2 = 0.999f;
	const float epsilon = 1e-16f;

	const float spawnRange = 10;
	const float halfSpawnRange = spawnRange * 0.5f;
	const float spawnRangeScalar = spawnRange / RAND_MAX;

	const float moveRange = 1;
	const float halfMoveRange = moveRange * 0.5f;
	const float moveRangeScalar = moveRange / RAND_MAX;

	const int maxEpisodes = 100000;
	const int maxSteps = 16;
	const int batchSize = 32;
	const int hiddenMemSize = 16;
	const int numInputs = 2;
	const int numOutputs = 1;

	const int inputSize = numInputs + hiddenMemSize;
	const int hiddenLayer1Size = 64;
	const int hiddenLayer2Size = 32;
	const int outputSize = numOutputs + hiddenMemSize;

	float x, y;
	float sqrDistances[maxSteps];
	float hiddenMemParam[hiddenMemSize];
	float inputs[inputSize * maxSteps];
	float hiddenLayer1[hiddenLayer1Size * maxSteps];
	float hiddenLayer2[hiddenLayer1Size * maxSteps];
	float outputs[outputSize * maxSteps];

	float hiddenLayer1Weight[hiddenLayer1Size * inputSize];
	float hiddenLayer2Weight[hiddenLayer2Size * hiddenLayer1Size];
	float outputLayerWeight[outputSize * hiddenLayer2Size];

	float hiddenMemParamGradients[hiddenMemSize];
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
	for (int i = 0; i < inputSize; ++i)
		for (int j = 0; j < hiddenLayer1Size; ++j)
			hiddenLayer1Weight[i * hiddenLayer1Size + j] = (i == j) * ((rand() & 2) - 1) + ((float)rand() / RAND_MAX * 0.2 - 0.1) / inputSize;

	for (int i = 0; i < hiddenLayer1Size; ++i)
		for (int j = 0; j < hiddenLayer2Size; ++j)
			hiddenLayer2Weight[i * hiddenLayer2Size + j] = (i == j) * ((rand() & 2) - 1) + ((float)rand() / RAND_MAX * 0.2 - 0.1) / hiddenLayer1Size;

	for (int i = 0; i < hiddenLayer2Size; ++i)
		for (int j = 0; j < outputSize; ++j)
			outputLayerWeight[i * outputSize + j] = (i == j) * ((rand() & 2) - 1) + ((float)rand() / RAND_MAX * 0.2 - 0.1) / hiddenLayer2Size;

	for (int i = 0; i < hiddenMemSize; ++i)
		hiddenMemParam[i] = ((float)rand() / RAND_MAX - 0.5) / hiddenMemSize;
	//memset(hiddenMemParam, 0, hiddenMemSize * sizeof(float));

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
		memset(hiddenMemParamGradients, 0, hiddenMemSize * sizeof(float));

		for (int batch = 0; batch < batchSize; ++batch)
		{
			float score = 0;

			/*x = rand() * spawnRangeScalar - halfSpawnRange;
			y = rand() * spawnRangeScalar - halfSpawnRange;*/

			for (int step = 0; step < maxSteps; ++step)
			{
				x = rand() & 1;
				y = rand() & 1;
				score += 2 * x + y;

				inputs[step * inputSize] = x;
				inputs[step * inputSize + 1] = y;
				sqrDistances[step] = score;
				//sqrDistances[step] = x + y;

				/**/if (step == 0)
				{
					// set the extra inputs to hiddenMemParam
					memcpy(inputs + numInputs, hiddenMemParam, hiddenMemSize * sizeof(float));
				}
				else
				{
					// set the extra inputs to the previous output
					memcpy(inputs + step * inputSize + numInputs, outputs + (step - 1) * outputSize + numOutputs, hiddenMemSize * sizeof(float));
				}
				
				// set the extra inputs to 0
				//memset(inputs + step * inputSize + numInputs, 0, hiddenMemSize * sizeof(float));
				//*(inputs + step * inputSize + numInputs) = 100;

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

				/**/if (step + 1 == maxSteps)
				{
					// set the extra output gradients to 0
					memset(outputGradients + step * outputSize + numOutputs, 0, hiddenMemSize * sizeof(float));
				}
				else
				{
					// set the extra output gradients to the previous input gradients
					memcpy(outputGradients + step * outputSize + numOutputs, inputGradients + (step + 1) * inputSize + numInputs, hiddenMemSize * sizeof(float));
				}

				// set the extra output gradients to 0
				//memset(outputGradients + step * outputSize + numOutputs, 0, hiddenMemSize * sizeof(float));

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

				cpuSgemmStridedBatched
				(
					true, false,
					inputSize, 1, hiddenLayer1Size,
					&alpha,
					hiddenLayer1Weight, hiddenLayer1Size, 0,
					hiddenLayer1Gradients + step * hiddenLayer1Size, hiddenLayer1Size, 0,
					&beta,
					inputGradients + step * inputSize, inputSize, 0,
					1
				);

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

			// extracting the last mem gradients
			cpuSaxpy(hiddenMemSize, 1, inputGradients + numInputs, hiddenMemParamGradients);
		}

		printf("averageError: %f\n", averageError / (maxSteps * batchSize));

		// adam
		/**/for (int i = 0; i < outputSize * hiddenLayer2Size; i++)
		{
			float gradient = outputLayerWeightGradients[i] / (maxSteps * batchSize);
			outputLayerWeightGradientMean[i] = beta1 * outputLayerWeightGradientMean[i] + (1 - beta1) * gradient;
			/*float delta = gradient - outputLayerWeightGradientMean[i];
			outputLayerWeightGradientVariance[i] = beta2 * outputLayerWeightGradientVariance[i] + (1 - beta2) * delta * delta;*/
			outputLayerWeightGradientVariance[i] = beta2 * outputLayerWeightGradientVariance[i] + (1 - beta2) * gradient * gradient;
			float correctedMean = outputLayerWeightGradientMean[i] / (1 - exponentiallyDecayedMean);
			float correctedVariance = outputLayerWeightGradientVariance[i] / (1 - exponentiallyDecayedVariance);
			 //outputLayerWeight[i] += learningRate * (gradient - correctedMean) * InvSqrt(correctedVariance + epsilon);
			outputLayerWeight[i] += learningRate * correctedMean * InvSqrt(correctedVariance + epsilon);
		}

		for (int i = 0; i < hiddenLayer2Size * hiddenLayer1Size; i++)
		{
			float gradient = hiddenLayer2WeightGradients[i] / (maxSteps * batchSize);
			hiddenLayer2WeightGradientMean[i] = beta1 * hiddenLayer2WeightGradientMean[i] + (1 - beta1) * gradient;
			/*float delta = gradient - hiddenLayer2WeightGradientMean[i];
			hiddenLayer2WeightGradientVariance[i] = beta2 * hiddenLayer2WeightGradientVariance[i] + (1 - beta2) * delta * delta;*/
			hiddenLayer2WeightGradientVariance[i] = beta2 * hiddenLayer2WeightGradientVariance[i] + (1 - beta2) * gradient * gradient;
			float correctedMean = hiddenLayer2WeightGradientMean[i] / (1 - exponentiallyDecayedMean);
			float correctedVariance = hiddenLayer2WeightGradientVariance[i] / (1 - exponentiallyDecayedVariance);
			//hiddenLayer2Weight[i] += learningRate * (gradient - correctedMean) * InvSqrt(correctedVariance + epsilon);
			hiddenLayer2Weight[i] += learningRate * correctedMean * InvSqrt(correctedVariance + epsilon);
		}

		for (int i = 0; i < hiddenLayer1Size * inputSize; i++)
		{
			float gradient = hiddenLayer1WeightGradients[i] / (maxSteps * batchSize);
			hiddenLayer1WeightGradientMean[i] = beta1 * hiddenLayer1WeightGradientMean[i] + (1 - beta1) * gradient;
			/*float delta = gradient - hiddenLayer1WeightGradientMean[i];
			hiddenLayer1WeightGradientVariance[i] = beta2 * hiddenLayer1WeightGradientVariance[i] + (1 - beta2) * delta * delta;*/
			hiddenLayer1WeightGradientVariance[i] = beta2 * hiddenLayer1WeightGradientVariance[i] + (1 - beta2) * gradient * gradient;
			float correctedMean = hiddenLayer1WeightGradientMean[i] / (1 - exponentiallyDecayedMean);
			float correctedVariance = hiddenLayer1WeightGradientVariance[i] / (1 - exponentiallyDecayedVariance);
			//hiddenLayer1Weight[i] += learningRate * (gradient - correctedMean) * InvSqrt(correctedVariance + epsilon);
			hiddenLayer1Weight[i] += learningRate * correctedMean * InvSqrt(correctedVariance + epsilon);
		}

		// sgd
		/*cpuSaxpy(outputSize * hiddenLayer2Size, learningRate / (maxSteps * batchSize), outputLayerWeightGradients, outputLayerWeight);
		cpuSaxpy(hiddenLayer2Size * hiddenLayer1Size, learningRate / (maxSteps * batchSize), hiddenLayer2WeightGradients, hiddenLayer2Weight);
		cpuSaxpy(hiddenLayer1Size * inputSize, learningRate / (maxSteps * batchSize), hiddenLayer1WeightGradients, hiddenLayer1Weight);
		cpuSaxpy(hiddenMemSize, learningRate / (batchSize), hiddenMemParamGradients, hiddenMemParam);
		*/
	}

	return 0;
}

/*
- Given Latent Representation, observation, and random vector, make an action	()
- Given Latent Representation and action, change the latent representation		()
- Given Latent Representation, predict reward									(sparse matrix)
- Given Latent Representation, predict next observation							(sparse matrix)

- inputs should be emply shells that accepts existing nodes
*/