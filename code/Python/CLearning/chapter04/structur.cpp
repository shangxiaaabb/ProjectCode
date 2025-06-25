// structur.cpp -- a simple structure
#include <iostream>
#include <string>
using namespace std;

struct inflatable   // structure declaration
{
    string name;
    float volume;
    double price;
};

int main()
{
    using namespace std;
    inflatable peo1[1]={
        {"aa", 1, 1}
    };
    cout<< peo1[0].name<<endl;
    inflatable guest =
    {
        "Glorious Gloria",  // name value
        1.88,               // volume value
        29.99               // price value
    }; 

    inflatable pal =
    {
        "Audacious Arthur",
        3.12,
        32.99
    };  // pal is a second variable of type inflatable
// NOTE: some implementations require using
// static inflatable guest =

    cout << "Expand your guest list with " << guest.name;
    cout << " and " << pal.name << "!\n";
// pal.name is the name member of the pal variable
    cout << "You can have both for $";
    cout << guest.price + pal.price << "!\n";
    // cin.get();
    return 0; 
}
