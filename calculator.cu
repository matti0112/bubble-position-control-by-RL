#include <device_launch_parameters.h>
#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

#define dimension 4
#define action_dim 2
#define pi 3.14159265358979324
#define threads 512


extern "C" __device__ double km(double t, double* x,double* action, double* cp, int eq_num) {

	double rx0 = 1.0 / x[0];
	double PA = action[0] * sin(2.0 * pi * x[1] + action[1]) * sin(2.0 * pi * t + action[1]);
	double PAT = action[0] * cp[17] * sin(2.0 * pi * x[1] + action[1]) * cos(2.0 * pi * t + action[1]);
	double GRADP = action[0] * cp[18] * cos(2.0 * pi * x[1] + action[1]) * sin(2.0 * pi * t + action[1]);
	double UAC = -action[0] * cp[16] * cos(2.0 * pi * x[1] + action[1]) * cos(2.0 * pi * t + action[1]);
	double N = (cp[0] + cp[1] * x[2]) * pow(rx0, cp[12])
		- cp[2] * (1 + cp[7] * x[2]) - cp[3] * rx0 - cp[4] * x[2] * rx0
		- 1.5 * (1.0 - cp[7] * x[2] * (1.0 / 3.0)) * x[2] * x[2]
		- (1.0 + cp[7] * x[2]) * cp[5] * PA - cp[6] * PAT * x[0]
		+ cp[8] * x[3] * x[3];                                    // Feedback term


	double D = x[0] - cp[7] * x[0] * x[2] + cp[4] * cp[7];
	double rD = 1.0 / D;

	double Fb1 = -cp[10] * x[0] * x[0] * x[0] * GRADP;          // Primary Bjerknes Force
	double Fd = -cp[11] * x[0] * (x[3] * cp[15] - UAC);

	if (eq_num == 0) {
		return x[2];
	}
	else if (eq_num == 1) {
		return x[3];
	}
	else if (eq_num == 2) {
		return N * rD;
	}
	else if (eq_num == 3) {
		return 3.0 * (Fb1 + Fd) * cp[9] * rx0 * rx0 * rx0 - 3.0 * x[2] * rx0 * x[3];
	}
}


// Runge-Kutta-Cash-Karp method
extern "C" __global__ void rkck(double* sol_v, double* cp, double* actioni) {
	printf("\ncuda started\n");
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid==0){
		for (int i=0; i<1024;i++){
			printf("%f, ",actioni[i]);
		}
	}
	double t = 0.0;
	double h = 1.0e-7;
	double tolerance = 1.0e-12;

	double k_matrix[6 * dimension];
	double x_v[dimension];
	double action[2];

	for (int i = 0; i < action_dim; i++) {
		action[i] = actioni[tid * action_dim + i];
	}

	for (int i = 0; i < dimension; i++) {
		x_v[i] = sol_v[tid * dimension + i];
	}

	while (t < 10.0) {
		// k1
		for (int i = 0; i < dimension; i++) {
			k_matrix[i] = h * km(t, x_v, action, cp, i % dimension);
		}

		// k2
		for (int i = 0; i < dimension; i++) {
			x_v[i] += 0.2 * k_matrix[i];
		}
		for (int i = 0; i < dimension; i++) { k_matrix[dimension + i] = h * km(t + 0.2 * h, x_v, action, cp, i % dimension); }

		// k3
		for (int i = 0; i < dimension; i++) {
			x_v[i] += -0.125 * k_matrix[i] + 0.225 * k_matrix[dimension + i];
		}
		for (int i = 0; i < dimension; i++) { k_matrix[2 * dimension + i] = h * km(t + 0.3 * h, x_v, action, cp, i % dimension); }

		// k4
		for (int i = 0; i < dimension; i++) {
			x_v[i] += 0.225 * k_matrix[i] - 1.125 * k_matrix[dimension + i] + 1.2 * k_matrix[2 * dimension + i];
		}
		for (int i = 0; i < dimension; i++) { k_matrix[3 * dimension + i] = h * km(t + 0.6 * h, x_v, action, cp, i % dimension); }

		// k5
		for (int i = 0; i < dimension; i++) {
			x_v[i] += -0.5037037037037037 * k_matrix[i] + 3.4 * k_matrix[dimension + i] - 3.792592592592593 * k_matrix[2 * dimension + i] + 1.296296296296296 * k_matrix[3 * dimension + i];
		}
		for (int i = 0; i < dimension; i++) { k_matrix[4 * dimension + i] = h * km(t + h, x_v, action, cp, i % dimension); }

		// k6
		for (int i = 0; i < dimension; i++) {
			x_v[i] += 0.2331995081018518 * k_matrix[i] - 2.158203125 * k_matrix[dimension + i] + 2.634186921296296 * k_matrix[2 * dimension + i] - 0.8959508825231481 * k_matrix[3 * dimension + i] + 0.061767578125 * k_matrix[4 * dimension + i];
		}
		for (int i = 0; i < dimension; i++) { k_matrix[5 * dimension + i] = h * km(t + 0.875 * h, x_v, action, cp, i % dimension); }

		double ratio = 100.0;
		// calculating estimated error and desired accuracy
		for (int i = 0; i < dimension; i++) {
			double ratio_i = (tolerance * (abs(sol_v[tid * dimension + i]) + h * abs(k_matrix[i]) + 10e-30)) / abs(-0.004293774801587311 * k_matrix[i] + 0.01866858609385785 * k_matrix[2 * dimension + i] - 0.03415502683080807 * k_matrix[3 * dimension + i] - 0.01932198660714286 * k_matrix[4 * dimension + i] + 0.03910220214568039 * k_matrix[5 * dimension + i]);
			if (ratio_i < ratio) {
				ratio = ratio_i;
			}
		}
		if (ratio < 1) {
			// reject
			h *= 0.9 * std::pow(ratio, 0.2);
			for (int i = 0; i < dimension; i++) {
				x_v[i] = sol_v[tid * dimension + i];
			}
		}
		else {
			// accept
			t += h;
			for (int i = 0; i < dimension; i++) {
				sol_v[tid * dimension + i] = sol_v[tid * dimension + i] + 0.09788359788359788 * k_matrix[i] + 0.4025764895330113 * k_matrix[2 * dimension + i] + 0.2104377104377105 * k_matrix[3 * dimension + i] + 0.2891022021456804 * k_matrix[5 * dimension + i];
				x_v[i] = sol_v[tid * dimension + i];
			}
			h *= 0.95 * std::pow(ratio, 0.25);
		}
		if (h > (10.0 - t)) {
			h = 10.0 - t;
		}
	}
}

extern "C" void call(double* state, double* cp, double* action) {
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);

	int size = threads;
	int threadsPerBlock = 32;
	int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
	rkck << <blocksPerGrid, threadsPerBlock >> > (state, cp, action);
	cudaEventRecord(stop);

	//calculate time
	float mili = 0.0;
	cudaEventElapsedTime(&mili, start, stop);
	printf("\nMeasured time: ");
	printf("%.6f [ms]\n----------------\n\n", mili);
}