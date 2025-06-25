// instr2.cpp -- reading more than one word with getline
#include <iostream>
#include <cstring>

int main()
{
    using namespace std;
    const int ArSize = 20;
    char name[ArSize];
    char dessert[ArSize];

    cout << "Enter your name:\n";
    cin.getline(name, ArSize, '!');  // reads through newline 通过 ! 进行终止输入
    cout << "Enter your favorite dessert:\n";
    cin.getline(dessert, ArSize);
    cout << "I have some delicious " << dessert;
    cout << " for you, " << name << ".\n";
    cout<< strlen(name);
    // cin.get();
    return 0; 
}
