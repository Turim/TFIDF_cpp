#include "tfidf.h"

#include <cassert>
#include <iostream>
#include <fstream>
#include <set>
#include <boost/tokenizer.hpp>
#include <sstream>

using namespace std;

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
int main()
{
	std::vector<std::vector<std::string>> inputData = loadData();
	tfidf ins(inputData);
	std::vector<std::vector<double>> mat = ins.weightMat;
}