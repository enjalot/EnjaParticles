#ifndef GUIPARSER_H
#define GUIPARSER_H

/** 
 * This class is the base class to write GUI informations parsers 
 * it provides all the basic functions to parse scripts comments 
 * in order to find informations related to the GUI you want to generate
 * for this WS Script
 * To write a parser for a new type of scripts, just extend that class 
 * and overload the parseProcedure function in order to match the language requirements
 */

//Needed spirit tools
#include <boost/spirit/core.hpp>
#include <boost/spirit/actor/push_back_actor.hpp>
#include <boost/spirit/actor/assign_actor.hpp>
#include <boost/spirit/actor/insert_key_actor.hpp>

//Other needed classes
#include <iostream>
#include <vector>
#include <utility>
#include <map>
#include <string> 
#include <stdlib.h>
#include <fstream>
#include <sstream>
#include <algorithm>

#include "GuiInformations.h"
#include "ParameterInformations.h"
#include "ImageInformations.h"
#include "ProcedureGuiInformations.h"

using namespace std;
using namespace boost::spirit;




/**
 * The class GuiParser that you'll have to extend in order to create new parsers
 */

class GuiParser {

 private:

	GuiInformations guiInformations;

  // This function needs to be overloaded in your subclasses to match procedures declaration
  virtual bool parseProcedure(char const* , map<string, vector<string> >&, string&, string& ) = 0;
  // Return true if the line starts the PROCEDURES section in the script file
  bool isBeginProcedures(char const*);
  // Return true if the line ends the PROCEDURES section in the script file
  bool isEndProcedures(char const*);
  // This function parses a #UI parameter informations line
  bool parseParameterInfos(char const*, ParameterInformations&);
  // This function parses a procedure presets line
  bool parseParametersPresets(char const* , vector<string>&);
  // This procedure is looking for IMAGE_WRITERS
  bool parseImageWriter(char const*, map<int,string>&);
  // This procedure is looking for the image informations in a procedure
  bool parseProcedureImage(char const*, ImageInformations&);
  bool parseProcedureGuiElements(char const*, GuiElementInformations&);
  bool parseProcedureReturn(char const*, string&);
/*

*/
  // This procedure displays formatted parameters informations
  //void showParameterInfos(parameterInfos);

 public:

  // This is the main function of the GuiParser class. Parses the entire file
  bool doParse(string fileToParse, GuiInformations& infos);
  	GuiParser(){};
};

#endif
