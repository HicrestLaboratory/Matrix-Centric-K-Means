

#include <iostream>
#include <string>
#include <unordered_map>

class ArgParse {
public:
    ArgParse(int argc, char* argv[]) {
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg.substr(0, 2) == "--") {
                if (i + 1 < argc && argv[i + 1][0] != '-') {
                    arguments_[arg.substr(2, arg.size())] = (argv[++i]);
                } else {
                    arguments_[arg.substr(2, arg.size())] = "";
                }
            }
        }
    }

    std::string getArg(const std::string& arg) const {
        if (arguments_.find(arg) != arguments_.end()) {
            return arguments_.at(arg);
        }
        return "";
    }

    bool isArgPresent(const std::string& arg) const {
        return arguments_.find(arg) != arguments_.end();
    }

    template <typename T>
    T getArgInt(const std::string& arg) const {
        return (T)std::atoi(getArg(arg).c_str());
    }

    float getArgFloat(const std::string& arg) const {
        return std::atof(getArg(arg).c_str());
    }

    std::unordered_map<std::string, std::string> arguments_;
};
