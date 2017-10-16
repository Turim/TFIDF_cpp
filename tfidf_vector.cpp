#include <cassert>
#include <iostream>
#include <fstream>
#include <set>
#include <boost/tokenizer.hpp>
#include <sstream>

#if 1 // print of the resulting matrix
# include <algorithm>
# include <iterator>
#endif

#if 1 // CSV related
# include <boost/algorithm/string.hpp>
# include <boost/lexical_cast.hpp>
# include <cstring>
# include <string>
#endif

#if 1
# include "opencv2/core.hpp"
# include "opencv2/ml.hpp"
#endif 

#include <algorithm>
#include <iterator>
#include <random>

#pragma comment(lib, "opencv_core330d.lib")
#pragma comment(lib, "opencv_ml330d.lib")

using namespace std;

class tfidf {
private:
	std::vector<std::vector<double>> dataMat; // converted bag of words matrix
	unsigned int nrow; // matrix row number
	unsigned int ncol; // matrix column number
	std::vector<std::vector<std::string>> rawDataSet; // raw data
	std::vector<std::string> vocabList; // all terms
	std::vector<int> numOfTerms; // used in tf calculation
	
	void createVocabList();
	inline std::vector<double> bagOfWords2VecMN(const std::vector<std::string> & inputSet);
	void vec2mat();
	inline std::vector<double> vec_sum(const std::vector<double>& a, const std::vector<double>& b);
	void calMat();

public:
	std::vector<std::vector<double>> weightMat; // TF-IDF weighting matrix
	tfidf(std::vector<std::vector<std::string>> & input):rawDataSet(input)
	{
		calMat();
	}
};

void tfidf::createVocabList()
{
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
	std::vector<double> returnVec(vocabList.size(), 0);
	for (std::string word : inputSet)
	{
		size_t idx = std::find(vocabList.begin(), vocabList.end(), word) - vocabList.begin();
		if (idx == vocabList.size())
			cout << "word: " << word << "not found" << endl;
		else
			returnVec.at(idx) += 1;
	}
	return returnVec;
}

void tfidf::vec2mat()
{
	int cnt(0);
	for (auto it = rawDataSet.begin(); it != rawDataSet.end(); ++ it)
	{
		cnt ++;
		cout << cnt << "\r";
		std::cout.flush();
		dataMat.push_back(bagOfWords2VecMN(*it));
		numOfTerms.push_back(it->size());
		it->clear();
	}
	cout << endl;
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

void tfidf::calMat()
{
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
}

namespace file_related
{
	std::string readFileText(const std::string & filename)
	{
		std::ifstream in(filename);
		std::string str((std::istreambuf_iterator<char>(in)),
			            std::istreambuf_iterator<char>());
		return str;
	}

	std::vector<std::string> textParse(const std::string & bigString)
	{
		std::vector<std::string> vec;
		boost::tokenizer<> tok(bigString);
		for(boost::tokenizer<>::iterator beg = tok.begin(); beg != tok.end(); ++ beg)
		{
		    vec.push_back(*beg);
		}
		return vec;
	}
}

std::vector<std::vector<std::string>> loadData()
{
	std::vector<std::vector<std::string>>  data;
	for (int i = 1; i != 50; ++i)
	{
		std::ostringstream ss;
		ss << "test_data/" << i << ".txt";
		std::string filename = ss.str();
		std::string str = file_related::readFileText(filename);
		if (str.empty())
			break;
		std::vector<std::string> wordList = file_related::textParse(str);
		data.push_back(wordList);
	}
	return data;
}

// when , is inside of " then it changes to ^
template<typename iterator>
void fold_quoted(iterator begin, iterator end)
{
	bool skip = true;
	for (; begin != end;++begin)
	{
		switch (*begin)
		{
		case ',':
			if (!skip)
				*begin = '^';
			break;
		case '"':
			skip = !skip;
			break;
		}
	}
}

template<typename iterator>
void unfold_quoted(iterator begin, iterator end)
{
	std::replace_if(begin, end, [](const char ch) -> bool { return ch == '^'; }, ',');
}

cv::Mat TfIdfWeights2Mat(const std::vector<std::vector<double>>& src)
{
	assert(!src.empty());
	const size_t srcSize = src.size();
	cv::Mat result(srcSize, src[0].size(), CV_32FC1);
	for (size_t ix = 0, ixMax = srcSize; ix < ixMax; ++ix)
	{
		const std::vector<double>& currentRow = src[ix];
		for (size_t jx = 0, jxMax = currentRow.size(); jx < jxMax; ++jx)
		{
			const double value = currentRow[jx];
			if (0. == value)
				continue;
			result.at<float>(ix, jx) = static_cast<float>(value);
		}
	}

	return result;
}

template<typename T, typename U>
void Shuffle(T& vec1, U& vec2)
{
	std::default_random_engine generator(time(NULL));
	assert(vec1.size() == vec2.size());

	for (std::vector<uint32_t>::size_type ix = 0; ix < vec1.size(); ++ix)
	{
		std::uniform_int_distribution<uint32_t> distribution(0, ix);
		std::vector<uint32_t>::size_type r = distribution(generator);
		std::swap(vec1[r], vec1[ix]);
		std::swap(vec2[r], vec2[ix]);
	}
}

int main(int argc, char** argv)
{
	std::vector<std::vector<std::string>> inputData;
	std::vector<vector<int>> responses;
	if ((argc >= 2) && (!strcmp("--test_csv", argv[1])))
	{
		assert(argc >= 3);
#if 0
		io::CSVReader<7, io::trim_chars<'\t', ' '>, io::no_quote_escape<','>, io::throw_on_overflow, io::empty_line_comment> in(argv[2]);
		in.read_header(io::ignore_extra_column, "url", "enumeration", "file", "text", "title", "url.1", "title_text");
		string url, enumeration, file, text, title, url1, title_text;
		while (in.read_row(url, enumeration, file, text, title, url1, title_text))
		{
			cout << url << "," << enumeration << "," << file << "," << text << "," << title << "," << url1 << "," << title_text << endl << endl;
		}
#endif
		// TODO: fix the csv parser & use it here.
		
		std::ifstream file(argv[2]);
		assert(!!file);
		string buff(1000, '\0');
		while (getline(file, buff))
		{
			string value(&buff[0]);
			// mask ','
			fold_quoted(value.begin(), value.end());
			vector<string> subStrings;
			boost::algorithm::split(subStrings, value, boost::is_any_of(","), boost::token_compress_off);
			assert(7 == subStrings.size());
			// unmask ','
			for_each(subStrings.begin(), subStrings.end(), [](string& val)
				{
					unfold_quoted(val.begin(), val.end());
				}
			);

			if (argc >= 4)
			{
				//  columnId. [0, 6]
				const size_t columnId = boost::lexical_cast<size_t>(argv[3]);
				assert(columnId < 7);
#if 0
				cout << subStrings[columnId] << endl;
#else
				inputData.emplace_back(vector<string>(1, subStrings[columnId]));
#endif
			}
			else
			{
				vector<string> current_data;
				for (size_t ix = 0; ix < 7; ++ix)
				{
#if 0
					cout << subStrings[ix];
					if (ix != 6)
					{
						cout << "|";
					}
#else
					current_data.emplace_back(subStrings[ix]);
#endif
				}
#if 0
				cout << endl;
#else
				inputData.emplace_back(current_data);
#endif
			}
		}

		// load the responses
		std::ifstream file_responses("mobile_text_classification_data/labels_final.csv");
		assert(!!file_responses);
		while (std::getline(file_responses, buff))
		{
			string value(&buff[0]);
			vector<string> subStrings;
			// mask ','
			fold_quoted(value.begin(), value.end());
			boost::algorithm::split(subStrings, value, boost::is_any_of(","), boost::token_compress_off);
			assert(8 == subStrings.size());
			vector<int> tempResponse;
			std::transform(subStrings.begin() + 1, subStrings.end(), back_inserter(tempResponse),
				[](string& value) -> int
				{
					// unmask ','
					unfold_quoted(value.begin(), value.end());
					return boost::lexical_cast<int>(value);
				}
			);
			responses.push_back(tempResponse);
		}
	}
	else
	{
		inputData = loadData();
	}

	assert(!inputData.empty());
	assert(inputData.size() == responses.size());
#if 0
	tfidf ins(inputData);
	std::vector<std::vector<double>> mat = ins.weightMat;
	for (size_t ix = 0, ixMax = mat.size(); ix < ixMax; ++ix)
	{
		copy(mat[ix].begin(), mat[ix].end(), ostream_iterator<double>(cout, " "));
		cout << endl;
	}
#endif


	// TODO: split into the training samples and the check samples
	const size_t trainCount = static_cast<size_t>(static_cast<float>(inputData.size()) * 0.7);
	const size_t testCount = inputData.size() - trainCount;
	cout << "trainCount = " << trainCount << "; testCount = " << testCount << endl;
	assert(trainCount > 0);
	assert(testCount > 0);

	vector<int> responsesVec;
	const size_t ResponseIndex = 1; // class to train
	transform(responses.begin(), responses.end(), back_inserter(responsesVec),
		[=](const vector<int>& data) -> int
		{
			assert(data.size() == 7);
			return data[ResponseIndex];
		}
	);
	assert(inputData.size() == responsesVec.size());

	Shuffle(inputData, responsesVec);

	using namespace cv;
	tfidf data(inputData);
	vector<vector<double>> trainDataVec(data.weightMat.begin(), data.weightMat.begin() + trainCount);
	vector<int> trainResponsesVec(responsesVec.begin(), responsesVec.begin() + trainCount);
	const Mat trainDataVecProxy = TfIdfWeights2Mat(trainDataVec);
	Ptr<ml::TrainData> trainData = ml::TrainData::create(trainDataVecProxy, ml::ROW_SAMPLE, trainResponsesVec);
	Ptr<ml::SVM> svm = ml::SVM::create();
	svm->setType(ml::SVM::C_SVC);
	svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6)); // 10^-6
	svm->setKernel(ml::SVM::LINEAR);
	svm->train(trainData);

	vector<vector<double>> testDataVec(data.weightMat.begin() + trainCount, data.weightMat.end());
	vector<int> testResponsesVec(responsesVec.begin() + trainCount, responsesVec.end());
#if 0
	vector<int> realResponsesVec(testResponsesVec.size(), 0);
#else
	Mat realResponsesVec(static_cast<int>(testResponsesVec.size()), 1, CV_32FC1);
#endif
	const Mat testDataVecProxy = TfIdfWeights2Mat(testDataVec);
	const float predictRc = svm->predict(testDataVecProxy, realResponsesVec);
	cout << "Testing the prediction..." << endl;

	size_t guessed = 0;
	size_t truePositive = 0, falsePositive = 0, trueNegative = 0, falseNegative = 0;
	for (size_t ix = 0, ixMax = testResponsesVec.size(); ix < ixMax; ++ix)
	{
		if (testResponsesVec[ix] == static_cast<int>(realResponsesVec.at<float>(ix)))
		{
			if (0 == testResponsesVec[ix])
			{
				++trueNegative;
			}
			else
			{
				++truePositive;
			}

			cout << "OK" << endl;
			++guessed;
		}
		else
		{
			cout << "FAIL (" << testResponsesVec[ix] << " against " << static_cast<int>(realResponsesVec.at<float>(ix)) << ")" << endl;
			if (static_cast<int>(realResponsesVec.at<float>(ix)) == 0)
			{
				++falseNegative;
			}
			else
			{
				++falsePositive;
			}
		}
	}

	cout << "----" << endl;
	cout << "Success rate: " << static_cast<float>(guessed) * 100 / testResponsesVec.size() << " %" << endl;
	cout << "Precision: " << static_cast<float>(truePositive) / (truePositive + falsePositive) << endl;
	cout << "Recall: " << static_cast<float>(truePositive) / (truePositive + falseNegative) << endl;

	return 0;
}