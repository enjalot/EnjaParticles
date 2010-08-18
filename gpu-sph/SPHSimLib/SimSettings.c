#include <map>
#include <iostream>

using namespace std;

class SimSettings
{
public:
	public SimSettings()
	{

	}

protected:
private:


	multimap<char,int> mymm;
	multimap<char,int>::iterator it;

	typedef multimap<string, string>::type SettingsMultiMap;
	SettingsMultiMap Settings;
};