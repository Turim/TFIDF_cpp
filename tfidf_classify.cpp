#include "tfidf.h"

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

#if 1 // vocabulary related
# include "json/json.h"
#endif 

#if 1
# include "opencv2/core.hpp"
# include "opencv2/ml.hpp"
#endif 

#include <boost/optional.hpp>

#include <algorithm>
#include <iterator>
#include <random>

#ifdef _DEBUG
# pragma comment(lib, "opencv_core330d.lib")
# pragma comment(lib, "opencv_ml330d.lib")
# pragma comment(lib, "jsoncppd.lib")
#else
# pragma comment(lib, "opencv_core330.lib")
# pragma comment(lib, "opencv_ml330.lib")
# pragma comment(lib, "jsoncpp.lib")
#endif

using namespace std;

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

enum class ClassifierType
{
	Unknown = 0,
	SVM,
	SVM_NO_TRAIN_1, // VLAD, vocabulary only usage
	LR
};

int main(int argc, char** argv)
try
{
	// Check the input params:
	if (argc < 2)
	{
		cout << "Usage: " << argv[0] << " <classifier_type> [params]" << endl;
		cout << "\twhere classifier_type = svm | svm_vocab | lr" << endl;
		cout << endl;
		return 1;
	}

	int classifierSubtype = -1; // TODO: boost::optional
	ClassifierType classifierType = ClassifierType::Unknown;
	if (!strcmp("svm", argv[1]) || !strcmp("svm_vocab", argv[1]))
	{
		if (!strcmp("svm", argv[1]))
			classifierType = ClassifierType::SVM;
		else if (!strcmp("svm_vocab", argv[1]))
			classifierType = ClassifierType::SVM_NO_TRAIN_1;
		else
			assert(!"Unknown SVC classifier");
		classifierSubtype = cv::ml::SVM::C_SVC;
		if ((argc >= 3) && (!strcmp("one_class", argv[2])))
			classifierSubtype = cv::ml::SVM::ONE_CLASS;
	}
	else if (!strcmp("lr", argv[1]))
		classifierType = ClassifierType::LR;
	else
		throw std::runtime_error(string("Unknown classifier type: \"") + argv[1] + "\"");
	

	// Stage I. Load the data
	std::vector<std::vector<std::string>> inputData;
	std::vector<vector<int>> responses;
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
	const string dataFile		= "mobile_classification_data/data_final.csv";
	const string responsesFile	= "mobile_classification_data/labels_final.csv";
	std::ifstream file(dataFile);
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

		//  columnId. [0, 6]
		const size_t columnId = 6;
		inputData.emplace_back(vector<string>(1, subStrings[columnId]));
	}

	// load the responses
	std::ifstream file_responses(responsesFile);
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

	assert(!inputData.empty());
	assert(inputData.size() == responses.size());

	// Stage II. Train the classifier
	const size_t trainCount = static_cast<size_t>(static_cast<float>(inputData.size()) * 0.7);
	const size_t testCount = inputData.size() - trainCount;
	cout << "trainCount = " << trainCount << "; testCount = " << testCount << endl;
	assert(trainCount > 0);
	assert(testCount > 0);

	// vocabulary
	vector<string> vocabulary;
	using namespace cv;
	Ptr<ml::SVM> svm = ml::SVM::create();
	assert(-1 != classifierSubtype); // TODO: boost::optional
	svm->setType(classifierSubtype);
	svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6)); // 10^-6
	svm->setKernel(ml::SVM::LINEAR);

	if (ClassifierType::SVM_NO_TRAIN_1 == classifierType)
	{
		{
			fstream vocab("vocabulary.json", ios_base::in);
			if (!!vocab)
			{
				Json::Value root;
				{
					Json::Reader jsonReader;
					jsonReader.parse(vocab, root, false);
				}
				for (size_t ix = 0; ix <= 5325; ++ix)
				{
					const double weight = root[boost::lexical_cast<string>(ix)]["idf"].asDouble();
					weight;
					string word = root[boost::lexical_cast<string>(ix)]["feature"].asString();
					vocabulary.emplace_back(std::move(word));
				}
			}
		}
	}

	vector<vector<int>> responsesVec;
	const size_t ClassesToTrain = 2; /* 7 */
	for (size_t classToTrain = 0; classToTrain < ClassesToTrain; ++classToTrain)
	{
		vector<int> responsesVecProxy;
		transform(responses.begin(), responses.end(), back_inserter(responsesVecProxy),
			[=](const vector<int>& data) -> int
			{
				assert(data.size() == 7);
				return data[classToTrain] == 0 ? 0 : (classToTrain+1);
			}
		);
		assert(inputData.size() == responsesVecProxy.size());
		responsesVec.emplace_back(responsesVecProxy);

		assert(responsesVec.size() > classToTrain);

		// train 'em all!!
		Shuffle(inputData, responsesVec[classToTrain]);
		tfidf data(inputData, vocabulary);
		vector<vector<double>> trainDataVec(data.weightMat.begin(), data.weightMat.begin() + trainCount);
		vector<int> trainResponsesVec(responsesVec[classToTrain].begin(), responsesVec[classToTrain].begin() + trainCount);
		const Mat trainDataVecProxy = TfIdfWeights2Mat(trainDataVec);
		Ptr<ml::TrainData> trainData = ml::TrainData::create(trainDataVecProxy, ml::ROW_SAMPLE, trainResponsesVec);
		svm->train(trainData);
	}

	// Stage III. Predict
	tfidf data(inputData, vocabulary);
	vector<vector<double>> testDataVec(data.weightMat.begin() + trainCount, data.weightMat.end());
	std::default_random_engine generator(time(NULL));
	assert(ClassesToTrain > 0);
	std::uniform_int_distribution<uint32_t> distribution(0, ClassesToTrain-1);
	std::vector<uint32_t>::size_type r = distribution(generator);
	vector<int> testResponsesVec(responsesVec[r].begin() + trainCount, responsesVec[r].end());
#if 0
	vector<int> realResponsesVec(testResponsesVec[r].size(), 0);
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
		const int assumedResponse = testResponsesVec[ix] * (r+1);
		const int realResponse = static_cast<int>(realResponsesVec.at<float>(ix));
		if (assumedResponse == realResponse)
		{
			if (0 == assumedResponse)
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
			cout << "FAIL (" << assumedResponse << " against " << realResponse << ")" << endl;
			if (realResponse == 0)
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
catch (std::exception& e)
{
	cout << "Exception has been caught: " << e.what() << endl;
	return 1;
}
catch (...)
{
	cout << "Unhandled exception: " << endl;
	return 1;
}