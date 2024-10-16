#pragma once
#include <vector>
#include <map>
#include <sstream>
#include <string>
#include <iomanip>
#include <chrono>
#include <fstream>
#include <iostream>

struct FuncTimer
{
private:
	std::map <std::string, double> timer, t1, t2;
	std::string active;
	double dt, total;
public:
	FuncTimer()
	{
		dt = total = 0;
	}
	void start(std::string s)
	{
		active = s;
		t1[s] = clock();
	}
	void end(std::string s)
	{
		if (t1.find(s) == t1.end())
		{
			std::cout << s + " trigger not started" << std::endl;
			return;
		}
		else
		{
			t2[s] = clock();
			dt = (t2[s] - t1[s]) / CLOCKS_PER_SEC;
			//if (s == active)
			{
				timer[s] += dt;
			}
		}
	}

	double get_whole_time(std::string s)
	{
		if (t1.find(s) == t1.end())
		{
			std::cout << s + " trigger not started" << std::endl;
			return 0.0;
		}
		else
		{
			t2[s] = clock();
			dt = (t2[s] - t1[s]) / CLOCKS_PER_SEC;
			return timer[s] + dt;
		}
	}

	double get_last_diff(std::string s)
	{
		if (t1.find(s) == t1.end())
		{
			std::cout << s + " trigger not started" << std::endl;
			return 0.0;
		}
		else
		{
			t2[s] = clock();
			dt = (t2[s] - t1[s]) / CLOCKS_PER_SEC;
			return dt = (t2[s] - t1[s]) / CLOCKS_PER_SEC;
		}
	}


	std::string get_info()
	{
		int n = int(timer.size());
		std::ostringstream oss;
		oss << "Calculation time in seconds. Number of cases: " << n << ".\n";

		for (auto& it : timer)
		{
			oss << it.first << ": " << it.second << std::endl;
		}
		return oss.str();
	}

	void show_info()
	{
		std::cout << get_info() << std::endl;
	}

	void write_info(std::string path = "report.dat")
	{
		std::ofstream file(path);
		file << get_info() << std::endl;
	}

};
