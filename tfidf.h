#ifndef TFIDF__TFIDF_H
#define TFIDF__TFIDF_H

#include <algorithm>
#include <cassert>
#include <iostream>
#include <iterator>
#include <set>
#include <string>
#include <vector>

class tfidf {
private:
	std::vector<std::vector<double>> dataMat; // converted bag of words matrix
	unsigned int nrow; // matrix row number
	unsigned int ncol; // matrix column number
	std::vector<std::vector<std::string>> rawDataSet; // raw data
	std::vector<std::string> vocabList; // all terms
	std::vector<int> numOfTerms; // used in tf calculation

private:
	void createVocabList();
	inline std::vector<double> bagOfWords2VecMN(const std::vector<std::string> & inputSet);
	void vec2mat();
	inline std::vector<double> vec_sum(const std::vector<double>& a, const std::vector<double>& b);
	void calMat();


public:
	std::vector<std::vector<double>> weightMat; // TF-IDF weighting matrix
	tfidf(const std::vector<std::vector<std::string>> & input) :rawDataSet(input)
	{
		calMat();
	}

	tfidf(const std::vector<std::vector<std::string>> & input, const std::vector<std::string>& vocabList)
		: rawDataSet(input), vocabList(vocabList)
	{
		calMat();
	}
};

inline void tfidf::createVocabList()
{
	assert(vocabList.empty());
	std::set<std::string> vocabListSet;
	for (std::vector<std::string> document : rawDataSet)
	{
		for (std::string word : document)
			vocabListSet.insert(word);
	}
	std::copy(vocabListSet.begin(), vocabListSet.end(), std::back_inserter(vocabList));
}

inline std::vector<double> tfidf::bagOfWords2VecMN(const std::vector<std::string> & inputSet)
{
	assert(!vocabList.empty());
	std::vector<double> returnVec(vocabList.size(), 0);
	for (std::string word : inputSet)
	{
		size_t idx = std::find(vocabList.begin(), vocabList.end(), word) - vocabList.begin();
		if (idx == vocabList.size())
			std::cout << "word: " << word << "not found" << std::endl;
		else
			returnVec.at(idx) += 1;
	}
	return returnVec;
}

inline void tfidf::vec2mat()
{
	int cnt(0);
	for (auto it = rawDataSet.begin(); it != rawDataSet.end(); ++it)
	{
		cnt++;
		std::cout << cnt << "\r";
		std::cout.flush();
		dataMat.push_back(bagOfWords2VecMN(*it));
		numOfTerms.push_back(it->size());
		it->clear();
	}
	std::cout << std::endl;
	ncol = dataMat[0].size();
	nrow = dataMat.size();
	rawDataSet.clear(); // release memory
}

inline std::vector<double> tfidf::vec_sum(const std::vector<double>& a, const std::vector<double>& b)
{
	assert(a.size() == b.size());
	std::vector<double> result;
	result.reserve(a.size());
	std::transform(a.begin(), a.end(), b.begin(),
		std::back_inserter(result), std::plus<double>());
	return result;
}

inline void tfidf::calMat()
{
	if (vocabList.empty())
		createVocabList();
	vec2mat();

	std::vector<std::vector<double>> dataMat2(dataMat);
	std::vector<double> termCount;
	termCount.resize(ncol);

	for (unsigned int i = 0; i != nrow; ++i)
	{
		for (unsigned int j = 0; j != ncol; ++j)
		{
			if (dataMat2[i][j] > 1) // only keep 1 and 0
				dataMat2[i][j] = 1;
		}
		termCount = vec_sum(termCount, dataMat2[i]); // no. of doc. each term appears
	}
	dataMat2.clear(); //release

	std::vector<double> row_vec;
	for (unsigned int i = 0; i != nrow; ++i)
	{
		for (unsigned int j = 0; j != ncol; ++j)
		{
			double tf = dataMat[i][j] / numOfTerms[i];
			double idf = log((double)nrow / (termCount[j]));
			row_vec.push_back(tf * idf); // TF-IDF equation
		}
		weightMat.push_back(row_vec);
		row_vec.clear();
	}
	nrow = weightMat.size();
	ncol = weightMat[0].size();
	std::cout << "calcMat: nrow = " << nrow << "; ncol = " << ncol << std::endl;
}




#endif // TFIDF__TFIDF_H
