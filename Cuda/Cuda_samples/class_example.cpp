#include<iostream>
#include<string.h>

using namespace std;



class Persona{
    
    //By default attribute are private
    int year;
    string name;

    public:
        // Costruttore con lista di inizializzazione
        Persona(const string& nome, int eta) : name("Sergio"), year(eta) {}
        // Method
        string to_string(){
            return "My name is " + name + std::to_string(year);
        }
        // Attribute 
        
};



int main(){

    cout<<"Programm started..."<<endl;

    Persona p1("luca" , 3);
   
    cout<<p1.to_string()<<endl;


    return 0;


}


