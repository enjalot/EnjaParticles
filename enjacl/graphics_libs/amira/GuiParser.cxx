bool GuiParser::doParse(string fileToParse, GuiInformations& infos){



	ifstream inputFileStream(fileToParse.c_str());

	guiInformations = infos;

// FLAGS
	bool insideProcedureSection = false;
	bool parsingProcedure = false;
	bool parsingProcedureParameters = false;
	bool parsingParametersPresets = false;
	bool parsingProcedureImage = false;
	bool parsingProcedureGuiElements = false;
	bool parsingProcedureReturn = false;

	string line;
	string lastAddedProcedure;
	string renderWindowName = "";
	vector<string> renderWindowNameList;
// Temporary DataStructure storing the parameters list of a procedure
// KEY (string): name of the procedure
// VALUE (vector<string>): orderd list of parameters

	map<string, vector<string> >  proceduresParams;

	map<string, ProcedureGuiInformations>::const_iterator proceduresIterator;
	map<int,string>::const_iterator writersIterator;

// Counter used to know how many parameters of the procedure have already been parsed
	int parametersCounter;
// Total of parameters to be parsed for a procedure
	int parametersCount;

//Reading the file line by line
	while(getline(inputFileStream, line)){

		if(insideProcedureSection){

			if(parsingProcedure){

				if(parsingProcedureImage){
					ImageInformations imageInformations;
					if(parseProcedureImage(line.c_str(),imageInformations)){
						guiInformations.procedures()[lastAddedProcedure].setImageInformations(imageInformations);
						parsingProcedureImage = false;
						parsingProcedureGuiElements = true;
						continue;
					}
					else{
						cout << "Warning in [" << lastAddedProcedure << "]: missing procedure image description." << endl;
		//parsingProcedure = false;
		//return false;
						parsingProcedureImage = false;
						parsingProcedureGuiElements = true;
					}  
				}

				if(parsingProcedureGuiElements){
					GuiElementInformations elementInfos;

					if(parseProcedureGuiElements(line.c_str(),elementInfos)){
						guiInformations.procedures()[lastAddedProcedure].addGuiElement(elementInfos);
					}
					else{
						if(guiInformations.procedures()[lastAddedProcedure].guiElements().size() == 0){
							cout << "Failure in [" << lastAddedProcedure << "]: no GUI elements specified." << endl;
							parsingProcedure = false;
							return false;
						}
						else{
							parsingProcedureGuiElements = false;
							parsingProcedureParameters = true;
						}
					}
				}

				if(parsingProcedureParameters){
					ParameterInformations paramInfos;
					if(parseParameterInfos(line.c_str(),paramInfos)){
						bool guiElementIdFound = false;
						for(vector<GuiElementInformations>::size_type j = 0; j < guiInformations.procedures()[lastAddedProcedure].guiElements().size(); ++j){
							if(guiInformations.procedures()[lastAddedProcedure].guiElements()[j].id() == paramInfos.guiElementId()){
								guiElementIdFound = true;
							}
						}
						if(guiElementIdFound){
							paramInfos.setName(proceduresParams[lastAddedProcedure][parametersCounter]);
							guiInformations.procedures()[lastAddedProcedure].addParameter(paramInfos);
							parametersCounter++;
							if(parametersCounter == parametersCount){
								parsingProcedureParameters = false;
								parsingProcedureReturn = true; 
								continue;
							}
						}
						else{
							cout << "Failure in [" << lastAddedProcedure << "]: guiElementId [" << paramInfos.guiElementId() << "] specified for parameter [" << 
								proceduresParams[lastAddedProcedure][parametersCounter] << "] doesn't exist." << endl;
							parsingProcedure = false;
							return false;
						}
					}
					else{
						cout << "Failure in [" << lastAddedProcedure << "]: missing some parameter(s) description." << endl;
						parsingProcedure = false;
						return false;
					}
				}


				if(parsingProcedureReturn){
					string procedureReturn;
					if(parseProcedureReturn(line.c_str(),procedureReturn)){
						guiInformations.procedures()[lastAddedProcedure].setHasReturn(true);
						guiInformations.procedures()[lastAddedProcedure].setReturnGuiElementId(procedureReturn);
						continue;
					}
					else {
						parsingProcedureReturn = false;
						parsingParametersPresets = true;	    
					}
				}


				if(parsingParametersPresets){
					vector<string> presets;
					if(parseParametersPresets(line.c_str(),presets)){
						if(presets.size() == parametersCount){
							guiInformations.procedures()[lastAddedProcedure].addPreset(presets);
						}
						else
							cout << "Failure in [" << lastAddedProcedure << "]: presets not saved due to presets count different from parameters count" << endl;
					}
					else{
						parsingParametersPresets = false;
						parsingProcedure = false;
					}
				}

			}
			else {
				if(parseProcedure(line.c_str(),proceduresParams,lastAddedProcedure, renderWindowName)) {
					ProcedureGuiInformations procInfos;
					procInfos.imageInformations().setInitialized(false);
					procInfos.setHasReturn(false);
					/*vector<string>::iterator renderWindowNameListIterator = find(renderWindowNameList.begin(), renderWindowNameList.end(), renderWindowName);
					if(renderWindowNameListIterator == renderWindowNameList.end()){*/
						procInfos.setRenderWindowName(renderWindowName);
					/*	if(renderWindowName != "") {
							renderWindowNameList.push_back(renderWindowName);
						}*/
						renderWindowName = "";
					/*}
					else{
						cout << "Failure in [" << lastAddedProcedure << "]: renderWindowName [" << renderWindowName << "] was used previously" << endl;
						parsingProcedure = false;
						return false;
					}*/
					guiInformations.addProcedure(lastAddedProcedure, procInfos);
					parsingProcedure = true;
					parsingProcedureImage = true;
					parametersCounter = 0;
					parametersCount = proceduresParams[lastAddedProcedure].size();
				}
			}
		}

		if(isBeginProcedures(line.c_str()))
			insideProcedureSection = true;

		if(isEndProcedures(line.c_str()))
			insideProcedureSection = false;

		parseImageWriter(line.c_str(), guiInformations.writers());

	}

	infos = guiInformations;

	return true;

}


bool 
GuiParser::isBeginProcedures(char const* str){

	return parse(str,str_p("#KW_PUBLIC")).full;

}

bool 
GuiParser::isEndProcedures(char const* str){

	return parse(str,str_p("##NO_PROCEDURES##")).full;

}

bool 
GuiParser::parseParameterInfos(char const* str, ParameterInformations& infos){
	bool minAssigned = false;
	bool maxAssigned = false;
	string guiElementId;
	double minValue;
	double maxValue;

	rule<> ui = (
		str_p("#UI") >> +blank_p >> str_p("PARAM") >> +blank_p >>
		(*alnum_p)[assign_a(guiElementId)] >> *blank_p >>
		!(str_p("MIN") >> +blank_p >> (real_p)[assign_a(minValue)])[assign_a(minAssigned,true)] >> *blank_p >>
		!(str_p("MAX") >> +blank_p >> (real_p)[assign_a(maxValue)])[assign_a(maxAssigned,true)] >> *blank_p 
		);
	bool res =  parse(str,ui).full;

	infos.setGuiElementId(guiElementId);
	infos.setMinValue(minValue);
	infos.setMaxValue(maxValue);
	if(!(minAssigned || maxAssigned))
		infos.setRangeFormat("noRange");
	if(maxAssigned && !minAssigned)
		infos.setRangeFormat("maxOnly");
	if(!maxAssigned && minAssigned)
		infos.setRangeFormat("minOnly");
	if(maxAssigned && minAssigned)
		infos.setRangeFormat("completeRange");


	return res;
}

bool 
GuiParser::parseProcedureGuiElements(char const* str, GuiElementInformations& infos){

	string id;
	string type;

	rule<> ui = (
		str_p("#UI") >> +blank_p >> str_p("ELEMENT") >> +blank_p >>
		(*alnum_p)[assign_a(id)] >> +blank_p >>
		(*alnum_p)[assign_a(type)] >> 
		*(+blank_p >> ch_p('"') >> (+(alnum_p || blank_p  || ch_p('.')))[push_back_a(infos.options())] >> ch_p('"')) >> *blank_p
		);
	bool res =  parse(str,ui).full;

	infos.setId(id);
	infos.setType(type);

	return res;
}

bool 
GuiParser::parseProcedureReturn(char const* str, string& returnGuiElementId){

	rule<> ui = (
		str_p("#UI") >> +blank_p >> str_p("RETURN") >> +blank_p >>
		(*alnum_p)[assign_a(returnGuiElementId)] >> *blank_p
		);
	bool res =  parse(str,ui).full;

	return res;
}

bool 
GuiParser::parseParametersPresets(char const* str, vector<string>& presets){
	rule<> ui = (
		str_p("#UI") >> +blank_p >> str_p("PRESET") >> 
		*(+blank_p >> (+alnum_p)[push_back_a(presets)]) >> *blank_p
		);
	bool res =  parse(str,ui).full;

	return res;
}

bool 
GuiParser::parseImageWriter(char const* str, map<int,string>& writers){
	int writerNumber;
	string writerName;
	rule<> ui = (
		str_p("#IMAGE_WRITER") >> uint_p[assign_a(writerNumber)] >> +blank_p >>  (+alnum_p)[assign_a(writerName)] >> *blank_p
		);
	bool res =  parse(str,ui).full;

	if(res)
		writers[writerNumber] = writerName;

	return res;
}


bool 
GuiParser::parseProcedureImage(char const* str, ImageInformations& infos){

	int height;
	int width;
	int posX;
	int posY;

	rule<> ui = (
		str_p("#UI IMAGE") >> 
		+blank_p >> (uint_p)[assign_a(height)] >> +blank_p >> (uint_p)[assign_a(width)] >> 
		+blank_p >> (uint_p)[assign_a(posX)] >> +blank_p >> (uint_p)[assign_a(posY)] >> 
		*blank_p
		);



	bool res =  parse(str,ui).full;
	if(res)
		infos.setHeight(height);
	infos.setWidth(width);
	infos.setPosX(posX);
	infos.setPosY(posY);
	infos.setInitialized(true);
	return res;
}


