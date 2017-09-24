# NB:
# tfidf_Eigen project is ignored at present as it has dependencies on the Eigen
# lyric_similarity project is also ignored as it doesn't related directly to the text similarity

cmake_minimum_required(VERSION 3.0)
project(tfidf_vector)


function(get_configuration_suffix outSuffix)
	set(suffix)
	if (MSVC_IDE)
		if (MSVC_VERSION LESS 1600)
			set (suffix "/$(ConfigurationName)")
		else ()
			set (suffix "/$(Configuration)")
		endif()
	else()
		set (suffix "")
	endif()

	set(${outSuffix} "${suffix}" PARENT_SCOPE)
endfunction()

function(get_configuration_runtime_dir outdir)	
	get_configuration_suffix(suffix)
	set(${outdir} "${CMAKE_BINARY_DIR}${suffix}" PARENT_SCOPE)
endfunction()




include_directories(
	../!external/boost
	./opencv/include
	./jsoncpp/include
)

link_directories(
	./opencv/x86/vc14/lib
	./jsoncpp/lib
)

add_executable(tfidf_vector tfidf_vector.cpp)
add_executable(tfidf_classify tfidf_classify.cpp)

get_configuration_runtime_dir(outdir)
add_custom_target(tfidf_data COMMAND
	${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/test_data ${outdir}/test_data
)
