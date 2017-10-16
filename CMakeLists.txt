# NB:
# tfidf_Eigen project is ignored at present as it has dependencies on the Eigen
# lyric_similarity project is also ignored as it doesn't related directly to the text similarity

cmake_minimum_required(VERSION 3.0)
project(tfidf_vector)

include_directories(
	../!external/boost
)

add_executable(tfidf_vector tfidf_vector.cpp)

