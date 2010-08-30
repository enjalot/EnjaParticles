#include "log.h"

using namespace isph;

std::string Log::outputFile;
std::ofstream Log::outputStream;
std::list<Log::Message> Log::messages;
void (*Log::receiver)(const Message&) = NULL;
Log::MessageType Log::logLevel = Log::DebugInfo;

void Log::Send(MessageType type, const std::string& text)
{
	// Skip logging if minimum level is higher
	if ((int)type < (int)logLevel) return;

	// log message
	Log::Message m;
	m.type = type;
	m.text = text;
	time_t t = time(NULL);
	m.when = localtime(&t);
	messages.push_back(m);

	// if enabled logging to file
	if(outputStream.is_open())
	{
		// print time
		char buffer[9];
		strftime(buffer, 9, "%X", m.when);
		outputStream << buffer;

		// print type
		switch(type)
		{
		case DebugInfo: outputStream << " | debug   | "; break;
		case Info:		outputStream << " | info    | "; break;
		case Warning:	outputStream << " | warning | "; break;
		case Error:		outputStream << " | ERROR   | "; break;
		default:		outputStream << " | user    | ";
		}

		// print description
		outputStream << text << std::endl;
	}
	
	// if user wants to catch messages, send it to him
	if(receiver)
		receiver(m);
}


void Log::SetOutput(const std::string& filename)
{
	LogDebug("Setting output log file: " + filename);
	outputFile = filename;

	// close old one
	if(outputStream.is_open())
		outputStream.close();

	// create file
	outputStream.open(filename.c_str());
	if(!outputStream.is_open())
		Send(Error, "Cannot create/open '" + filename + "' for logging");
}

void Log::SetLevel(MessageType level) 
{
    logLevel = level;
}


Log::~Log()
{
	if(outputStream.is_open())
		outputStream.close();
}
