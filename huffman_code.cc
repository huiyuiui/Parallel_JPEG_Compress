#include "huffman_code.h"

using namespace std;

HuffmanNode* buildHuffmanTree(const unordered_map<int, int>& frequencies) {
    priority_queue<HuffmanNode*, vector<HuffmanNode*>, Compare> pq;

    for (const auto& pair : frequencies) {
        pq.push(new HuffmanNode(pair.first, pair.second));
    }

    while (pq.size() > 1) {
        HuffmanNode* left = pq.top();
        pq.pop();
        HuffmanNode* right = pq.top();
        pq.pop();

        
        HuffmanNode* merged = new HuffmanNode(-1, left->freq + right->freq);

        merged->left = (left->freq >= right->freq) ? right : left;
        merged->right = (left->freq >= right->freq) ? left : right;

        pq.push(merged);
    }
    HuffmanNode* root = pq.top();
    return root; 
}


void generateCodes(HuffmanNode* node, const string& prefix, unordered_map<int, string>& codebook) {
    if (!node) return;

    if (node->value != -1) {
        codebook[node->value] = prefix;
    } 
    else {
        generateCodes(node->left, prefix + "0", codebook);
        generateCodes(node->right, prefix + "1", codebook);
    }
}


unordered_map<int, int> calculateFrequencies(const int* data, int size) {
    unordered_map<int, int> frequencies;

    for (int i = 0; i < size; ++i) {
        frequencies[data[i]]++;
    }

    return frequencies;
}


string encodeData(const int* data, int size, const unordered_map<int, string>& codebook) {
    string encodedData;
    for (int i = 0; i < size; ++i) {
        encodedData += codebook.at(data[i]);
    }
    return encodedData;
}
// string encodeData(const int* data, int size, const unordered_map<int, string>& codebook) {
    
//     vector<string> threadResults(omp_get_max_threads());

//     #pragma omp parallel
//     {
//         int tid = omp_get_thread_num();
//         stringstream localStream;

//         #pragma omp for
//         for (int i = 0; i < size; ++i) {
//             localStream << codebook.at(data[i]);
//         }

//         threadResults[tid] = localStream.str();
//     }

//     string encodedData;
//     for (const auto& partialResult : threadResults) {
//         encodedData += partialResult;
//     }

//     return encodedData;
// }

pair<string, unordered_map<int, string>> huffman_encode(const int* data, int size) {
    auto frequencies = calculateFrequencies(data, size);
   
    HuffmanNode* root = buildHuffmanTree(frequencies);
    unordered_map<int, string> codebook;
    generateCodes(root, "", codebook);

    string encodedData = encodeData(data, size, codebook);

    return {encodedData, codebook};
}

int* huffman_decode(const string& encodedData, const unordered_map<int, string>& codebook) {
    
    unordered_map<string, int> reverseCodebook;
    for (const auto& pair : codebook) {
        reverseCodebook[pair.second] = pair.first;
    }

    vector<int> decodedVector;
    string currentCode;

    for (char bit : encodedData) {
        currentCode += bit;
        if (reverseCodebook.find(currentCode) != reverseCodebook.end()) {
            decodedVector.push_back(reverseCodebook[currentCode]);
            currentCode.clear();
        }
    }

    int size = decodedVector.size();
    int *decodedValues = new int[size];
    for (int i = 0; i < size; ++i) {
        decodedValues[i] = decodedVector[i];
    }

    return decodedValues;
}
