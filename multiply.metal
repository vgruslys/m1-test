#define N 256

kernel void multiply_matrix_and_vector(
                       device const float* matrix,
                       device const float* vector,
                       device float* result,
                       uint index [[thread_position_in_grid]])
{
    float total = 0.0;
    int offset = index * N;

    for (int i = 0; i < N; ++i) {
        total += matrix[offset + i];
    }

    result[index] = total;
}

/*
kernel void multiply_matrix_and_vector(
                       device const float* matrix,
                       device const float* vector,
                       device float* result,
                       uint index [[thread_position_in_grid]])
{
}
*/
