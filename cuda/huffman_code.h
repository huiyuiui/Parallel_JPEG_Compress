// huffman_code.h
#ifndef HUFFMAN_CODE_H
#define HUFFMAN_CODE_H
#include <iostream>
#include <unordered_map>
#include <queue>
#include <string>
#include <sstream>
using namespace std;

struct HuffmanNode {
    int value;
    int freq;
    HuffmanNode* left;
    HuffmanNode* right;

    HuffmanNode(int v, int f) : value(v), freq(f), left(nullptr), right(nullptr) {}
};

struct Compare {
    bool operator()(HuffmanNode* a, HuffmanNode* b) {
        
        return a->freq > b->freq;
    }
};

pair<string, unordered_map<int, string>> huffman_encode(const int* data, int size);

int* huffman_decode(const string& encodedData, const unordered_map<int, string>& codebook, int outSize);

HuffmanNode* buildHuffmanTree(const unordered_map<int, int>& frequencies);

void generateCodes(HuffmanNode* node, const string& prefix, unordered_map<int, string>& codebook);

unordered_map<int, int> calculateFrequencies(const int* data, int size) ;

string encodeData(const int* data, int size, const unordered_map<int, string>& codebook);

vector<pair<int, int>> rle_compress(const int* data, int size);

vector<int> rle_decompress(const vector<pair<int, int>>& compressedData, size_t targetLength);

#endif // HUFFMAN_CODE_H