// strtype1.cpp -- using the C++ string class
#include <iostream>
#include <string>               // make string class available

using namespace std;

int main()
{
    char charr1[20];            // create an empty array
    char charr2[20] = "jaguar"; // create an initialized array
    string str1;                // create an empty string object
    string str2 = "panther";    // create an initialized string

//     string str_1 = "huang";
//     string str_2 {"jie"};
//     cout<< "str_1:" << str_1<< " add str_2:"<< str_2<< "="<< str_1+ str_2<< endl;

    cout << "Enter a kind of feline: ";
    cin >> charr1;
    cout << "Enter another kind of feline: ";
    cin >> str1;                // use cin for input
    cout << "Here are some felines:\n";
    cout << charr1 << " " << charr2 << " "
         << str1 << " " << str2 // use cout for output
         << endl;
    cout << "The third letter in " << charr2 << " is "
         << charr2[2] << endl;
    cout << "The third letter in " << str2 << " is "
         << str2[2] << endl;    // use array notation
    // cin.get();
	// cin.get();

    return 0; 
}
