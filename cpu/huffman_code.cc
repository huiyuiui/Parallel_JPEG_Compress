#include "huffman_code.h"

using namespace std;

vector<pair<int, int>> rle_compress(const int* data, int size) {
    vector<pair<int, int>> compressedData;

    int zeroCount = 0;

    for (int i = 0; i < size; ++i) {
        if (data[i] == 0) {
            zeroCount++;
        } else {
            compressedData.push_back({zeroCount, data[i]});
            zeroCount = 0;
        }
    }

   
    if (zeroCount > 0) {
        compressedData.push_back({-1, -1}); // EOB 標記
    }

    return compressedData;
}

int* rle_decompress(const vector<pair<int, int>> compressedData, int targetLength) {
    int* decompressedData = new int[targetLength];
    int currentIndex = 0;

    for (const auto& pair : compressedData) {
        if (pair.first == -1 && pair.second == -1) {
            while (currentIndex < targetLength) {
                decompressedData[currentIndex++] = 0;
            }
            break;
        } else {
            for (int i = 0; i < pair.first && currentIndex < targetLength; ++i) {
                decompressedData[currentIndex++] = 0;
            }
            decompressedData[currentIndex++] = pair.second;
        }
    }

    return decompressedData;
}


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

    if (node->left == NULL && node->right == NULL) {
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
        auto it = codebook.find(data[i]);
        if (it != codebook.end()) {
            encodedData += it->second;
        } 
        else {
            cerr << "Error: Value " << data[i] << " not found in codebook." << endl;
            exit(EXIT_FAILURE);
        }
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

    auto compressedData = rle_compress(data, size);
    vector<int> rleFlattened;
    for (const auto& pair : compressedData) {
        rleFlattened.push_back(pair.first);
        rleFlattened.push_back(pair.second);
    }
    auto frequencies = calculateFrequencies(rleFlattened.data(), rleFlattened.size());
    
    HuffmanNode* root = buildHuffmanTree(frequencies);
    unordered_map<int, string> codebook;
    generateCodes(root, "", codebook);

    string encodedData = encodeData(rleFlattened.data(), rleFlattened.size(), codebook);

    return {encodedData, codebook};
}

int* huffman_decode(const string& encodedData, const unordered_map<int, string>& codebook, int outSize) {
    
    unordered_map<string, int> reverseCodebook;
    for (const auto& pair : codebook) {
        reverseCodebook[pair.second] = pair.first;
    }

    // Step 2: Huffman 解碼展平數據
    vector<int> rleFlattened;
    string currentCode;
    for (char bit : encodedData) {
        currentCode += bit;
        if (reverseCodebook.find(currentCode) != reverseCodebook.end()) {
            rleFlattened.push_back(reverseCodebook[currentCode]);
            currentCode.clear();
        }
    }

    vector<pair<int, int>> compressedData;
    for (int i = 0; i < static_cast<int>(rleFlattened.size()); i += 2) {
        compressedData.push_back({rleFlattened[i], rleFlattened[i + 1]});
    }

    return rle_decompress(compressedData, outSize);
}