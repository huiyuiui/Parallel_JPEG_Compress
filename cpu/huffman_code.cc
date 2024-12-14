#include "huffman_code.h"

using namespace std;

// vector<pair<int, int>> rle_compress(const int* data, int size) {
//     vector<pair<int, int>> compressedData;

//     int zeroCount = 0;

//     for (int i = 0; i < size; ++i) {
//         if (data[i] == 0) {
//             zeroCount++;
//         } else {
//             compressedData.push_back({zeroCount, data[i]});
//             zeroCount = 0;
//         }
//     }

   
//     if (zeroCount > 0) {
//         compressedData.push_back({-1, -1}); // EOB 標記
//     }

//     return compressedData;
// }

vector<pair<int, int>> rle_compress(const int* data, int size) {
    vector<pair<int, int>> compressedData;
    int CHUNK = 16;
    __m512i zeroVec = _mm512_set1_epi32(0);

    int num_threads = omp_get_max_threads();
    vector<vector<pair<int, int>>> allCompressed(num_threads);
    vector<int> trailingZeroCounts(num_threads, 0);

    #pragma omp parallel
    {
        int threadId = omp_get_thread_num();
        vector<pair<int, int>>& localCompressed = allCompressed[threadId];
        int zeroCount = 0;

        #pragma omp for schedule(static)
        for (int i = 0; i <= size + CHUNK; i += CHUNK) {
            __m512i v = _mm512_loadu_si512((const __m512i*)(data + i));
            __mmask16 cmp_mask = _mm512_cmpeq_epi32_mask(v, zeroVec);

            for (int j = 0; j < CHUNK; j++) {
                if ((cmp_mask >> j) & 1) {
                    zeroCount++;
                } else {
                    localCompressed.emplace_back(zeroCount, data[i + j]);
                    zeroCount = 0;
                }
            }
        }

        #pragma omp for schedule(static)
        for (int i = (size / CHUNK) * CHUNK; i < size; i++) {
            if (data[i] == 0) {
                zeroCount++;
            } else {
                localCompressed.emplace_back(zeroCount, data[i]);
                zeroCount = 0;
            }
        }

        trailingZeroCounts[threadId] = zeroCount;
    }

    int accumulatedZeroCount = 0;
    for (int t = 0; t < num_threads; t++) {
        if (accumulatedZeroCount > 0 && !allCompressed[t].empty()) {
            allCompressed[t][0].first += accumulatedZeroCount;
            accumulatedZeroCount = 0;
        }

        compressedData.insert(compressedData.end(), allCompressed[t].begin(), allCompressed[t].end());

        accumulatedZeroCount += trailingZeroCounts[t];
    }

    if (accumulatedZeroCount > 0) {
        compressedData.emplace_back(-1, -1);
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


// unordered_map<int, int> calculateFrequencies(const int* data, int size) {
//     unordered_map<int, int> frequencies;

//     for (int i = 0; i < size; ++i) {
//         frequencies[data[i]]++;
//     }

//     return frequencies;
// }

unordered_map<int, int> calculateFrequencies(const int* data, int size) {
    int total_num = omp_get_max_threads();
    vector<unordered_map<int, int>> threadFrequencies(total_num);

    #pragma omp parallel
    {
        int threadId = omp_get_thread_num();
        for (int i = threadId; i < size; i += total_num) {
            threadFrequencies[threadId][data[i]]++;
        }
    }

    unordered_map<int, int> frequencies;
    for (const unordered_map<int, int>& threadFrequency : threadFrequencies) {
        for (const pair<int, int>& pair : threadFrequency) {
            frequencies[pair.first] += pair.second;
        }
    }

    return frequencies;
}

// string encodeData(const int* data, int size, const unordered_map<int, string>& codebook) {
//     string encodedData;

//     for (int i = 0; i < size; ++i) {
//         auto it = codebook.find(data[i]);
//         if (it != codebook.end()) {
//             encodedData += it->second;
//         } 
//     }
    
//     return encodedData;
// }
string encodeData(const int* data, int size, const unordered_map<int, string>& codebook) {
    int total_num = omp_get_max_threads();

    // Precompute chunk boundaries for each thread
    vector<int> chunk_starts(total_num), chunk_sizes(total_num);
    int base_size = size / total_num;
    int remainder = size % total_num;
    int offset = 0;

    for (int t = 0; t < total_num; t++) {
        int chunk_size = base_size + (t < remainder ? 1 : 0);
        chunk_starts[t] = offset;
        chunk_sizes[t] = chunk_size;
        offset += chunk_size;
    }

    vector<string> threadResults(total_num);

    #pragma omp parallel
    {
        int threadId = omp_get_thread_num();
        int start = chunk_starts[threadId];
        int end   = start + chunk_sizes[threadId];

        string localEncoded;
        localEncoded.reserve((end - start) * 8);

        for (int i = start; i < end; i++) {
            auto it = codebook.find(data[i]);
            if (it != codebook.end()) {
                localEncoded += it->second;
            }
        }

        threadResults[threadId] = move(localEncoded);
    }

    string encodedData;
    size_t totalLength = 0;
    for (auto &res : threadResults) totalLength += res.size();
    encodedData.reserve(totalLength);

    for (auto &res : threadResults) {
        encodedData += move(res);
    }

    return encodedData;
}

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