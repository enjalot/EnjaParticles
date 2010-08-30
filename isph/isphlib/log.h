#ifndef ISPH_LOG_H
#define ISPH_LOG_H

#include <string>
#include <list>
#include <fstream>
#include <ctime>

namespace isph {

	/*!
	 *	\brief	System for logging runtime library errors, warnings, etc.
	 *
	 *	This is the class that catches every (debug) info, warning, error, or user message and
	 *	processes it. Messages can be written to files and/or forwarded to user function for
	 *	processing messages.
	 */
	class Log
	{
	public:
		Log(){}
		~Log();

		/*!
		 *	\brief	Types of logged messages.
		 */
		enum MessageType
		{
			DebugInfo,
			Info,
			Warning,
			Error,
			User
		};

		/*!
		 *	\brief	Logged message of type MessageType with some info.
		 */
		struct Message
		{
			MessageType type;
			std::string text;
			tm* when;
		};

		/*!
		 *	\brief	Open file where to log the messages.
		 */
		static void SetOutput(const std::string& filename);

		/*!
		 *	\brief	Get the filename of log.
		 */
		static const std::string& Output() { return outputFile; }

		/*!
		 *	\brief	Add a new message to log.
		 *	\param	type	Type of the new message.
		 *	\param	text	Message.
		 *	\remarks Message is directly passes to user reciever if one is set.
		 */
		static void Send(MessageType type, const std::string& text);

		/*!
		 *	\brief	Get the list of all of the logged messages.
		 */
		static const std::list<Message>& Messages() { return messages; }

		/*!
		 *	\brief	Get the last logged message.
		 */
		static const Message& LastMessage() { return messages.back(); }

		/*!
		 *	\brief	Set user function to receive newly sent messages to logger.
		 */
		static void SetUserReceiver(void (*userFunc)(const Message&)) { receiver = userFunc; }

		/*!
		 *	\brief	Set minimum level of message to be logged.
		 */
		static void SetLevel(MessageType level);

	private:

		static std::string outputFile;
		static std::ofstream outputStream;
		static std::list<Message> messages;
		static MessageType logLevel;
		static void (*receiver)(const Message&);
	};

	// simple debug macro
	#define LogDebug(DESC) Log::Send(Log::DebugInfo, DESC)
}

#endif
