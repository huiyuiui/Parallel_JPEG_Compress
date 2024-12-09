#ifndef _DCT_H_
#define _DCT_H_

float* DCT(float *input, int height, int width);
float* iDCT(int *input, int height, int width);

#endif

// DCT:
// for (int u = 0; u < N; u++) {
//     alpha_u = (u == 0) ? 1.0/sqrt(2) : 1;
//     for (int v = 0; v < N; v++) {
//         alpha_v = (v == 0) ? 1.0/sqrt(2) : 1;
        
//         output[u * stride + v] = 0;
//         for (int x = 0; x < N; x++) {
//             for (int y = 0; y < N; y++) {
//                 output[u * stride + v] += input[u * stride + v] * 
//                                 cos((2*x+1)*u*M_PI/(2.0*N)) * 
//                                 cos((2*y+1)*v*M_PI/(2.0*N));
//             }
//         }
//         output[u * stride + v] *= 2.0 / N * alpha_u * alpha_v;
//     }
// }

// iDCT:
// for (int x = 0; x < N; x++) {
//     for (int y = 0; y < N; y++) {
//         output[x][y] = 0;
        
//         for (int u = 0; u < N; u++) {
//             alpha_u = (u == 0) ? 1.0/sqrt(2) : 1;
//             for (int v = 0; v < N; v++) {
//                 alpha_v = (v == 0) ? 1.0/sqrt(2) : 1;
//                 output[x][y] += alpha_u * alpha_v * input[u][v] * 
//                                 cos((2*x+1)*u*M_PI/(2.0*N)) * 
//                                 cos((2*y+1)*v*M_PI/(2.0*N));
//             }
//         }
//         output[x][y] *= 2.0 / N;
//     }
// }