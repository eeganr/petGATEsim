#include <pybind11/pybind11.h>
#include <iostream>
#include <vector>
#include <string>

using namespace std;

int main(int argc, char *argv[]) {
    vector<string> msg {"Hello", "C++", "World", "from", "VS Code", "and the C++ extension!", "meow"};

    for (const string& word : msg)
    {
        cout << word << " ";
    }
    cout << endl;
    return 0;
}