import json


class ObjectIO:
    def __init__(self, dataDict):
        self.Object = json.load(dataDict)

    def GetConfigSetting(self, keyWord):
        try:
            if keyWord in self.Object: 
                return self.Object[keyWord]
            else:
                raise KeyError
        except:
            print("KeyError:", keyWord, "is not included in the ConfigSettings!")
    
    def GetEssential(self, nb, keyWord):
        try:
            if keyWord in self.Object[nb]: 
                return self.Object[nb][keyWord]
            else:
                raise KeyError
        except:
            print("KeyError:", keyWord, "is not included in the data dictionary!")
    
    def GetAlternative(self, nb, keyWord):
        if keyWord in self.Object[nb]:
            return self.Object[nb][keyWord]
        else:
            return []
    
