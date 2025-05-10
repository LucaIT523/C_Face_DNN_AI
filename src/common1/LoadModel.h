#ifndef LOAD_MODEL_InferenceH
#define LOAD_MODEL_InferenceH
//---------------------------------------------------------------------------
#include <iostream>                    // std::cout
#include <fstream>                     // std::ofstream
#include <filesystem>                  // std::filesystem
#include <string>                      // std::string
#include <vector>                      // std::vector
#include <random>                      // std::random_device
#include <cstdlib>                     // std::srand, std::rand
#include "FaceSuperEng.hpp"
#include "FaceParsing.hpp"

//using namespace std;


// ----------------------------------------------------------------------
// 
// ----------------------------------------------------------------------
int			//. Return - 0 if success.
SetLoadModelData(
	FaceSuperEng&	p_FaceSuperEng
,	std::string		p_strModelFolder
);

// ----------------------------------------------------------------------
// 
// ----------------------------------------------------------------------
int			//. Return - 0 if success.
SetLoadModelData_FaceParsing(
	CFaceParsing&	p_FaceParsing
,	std::string		p_strModelFolder
);

// ----------------------------------------------------------------------
// 
// ----------------------------------------------------------------------
int			//. Return - Count.
GetModelCount(
	std::string		p_strModelFolder
);


// ----------------------------------------------------------------------
// 
// ----------------------------------------------------------------------
int			//. Return - 0 if success.	
GetModelInfo(	
	std::vector<std::string>&	p_strKeyName
,	std::vector<torch::Tensor>&	p_tsData
,	std::string					p_strModelFolder
);





#endif // LOAD_MODEL_InferenceH
