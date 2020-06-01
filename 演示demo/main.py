import wx
import numpy as np_main
np_main.set_printoptions(suppress=True)
import random
import pandas as iofunc
import csv
import os

import dota_predict

class MyFrame(wx.Frame):


    def __init__(self):
        wx.Frame.__init__ (self,None,-1,'Doat2 Prediction System', size=(800, 600))
        panel = wx.Panel (self)
        list1 = ["antimage","axe","bane","bloodseeker","crystal_maiden","drow_ranger","earthshaker",    #7
                 "juggernaut","mirana","morphling","nevermore","phantom_lancer","puck","pudge","razor","sand_king",  #16
                 "storm_spirit","sven","tiny","vengefulspirit","windrunner","zuus","kunkka","None","lina",     #25
                 "lion","shadow_shaman","slardar","tidehunter","witch_doctor","lich","riki","enigma",       #33
                 "tinker","sniper","necrolyte","warlock","beastmaster","queenofpain","venomancer","faceless_void", #41
                 "skeleton_king","death_prophet","phantom_assassin","pugna","templar_assassin","viper","luna",   #48
                 "dragon_knight","dazzle","rattletrap","leshrac","furion","life_stealer","dark_seer","clinkz",   #56
                 "omniknight","enchantress","huskar","night_stalker","broodmother","bounty_hunter","weaver",    #63
                 "jakiro","batrider","chen","spectre","ancient_apparition","doom_bringer","ursa","spirit_breaker",  #71
                 "gyrocopter","alchemist","invoker","silencer","obsidian_destroyer","lycan","brewmaster","shadow_demon", #79
                 "lone_druid","chaos_knight","meepo","treant","ogre_magi","undying","rubick","disruptor","nyx_assassin",  #88
                 "naga_siren","keeper_of_the_light","wisp","visage","slark","medusa","troll_warlord","centaur",   #96
                 "magnataur","shredder","bristleback","tusk","skywrath_mage","abaddon","elder_titan","legion_commander",  #104
                 "techies","ember_spirit","earth_spirit","abyssal_underlord","terrorblade","phoenix","oracle",    #111
                 "winter_wyvern","arc_warden"]   #113
        #print(list1[1])

        #self.label = wx.StaticText(panel,-1,)
        wx.StaticText(panel, -1, "Choose your opponent's hero:",(50, 20))

        self.listbox1 = wx.Choice(panel, -1, (50, 50), (100, 40), list1)
        self.listbox2 = wx.Choice (panel, -1, (200, 50), (100, 40), list1)
        self.listbox3 = wx.Choice (panel, -1, (350, 50), (100, 40), list1)
        self.listbox4 = wx.Choice (panel, -1, (500, 50), (100, 40), list1)
        self.listbox5 = wx.Choice (panel, -1, (650, 50), (100, 40), list1)

        wx.StaticText (panel, -1, "Choose your hero:", (50, 100))

        self.listbox6 = wx.Choice (panel, -1, (50, 130), (100, 40), list1)
        self.listbox7 = wx.Choice (panel, -1, (200, 130), (100, 40), list1)
        self.listbox8 = wx.Choice (panel, -1, (350, 130), (100, 40), list1)
        self.listbox9 = wx.Choice (panel, -1, (500, 130), (100, 40), list1)
        self.listbox10 = wx.Choice (panel, -1, (650, 130), (100, 40), list1)

        self.listbox1.Bind(wx.EVT_CHOICE,self.Hero1)
        self.listbox2.Bind (wx.EVT_CHOICE, self.Hero2)
        self.listbox3.Bind (wx.EVT_CHOICE, self.Hero3)
        self.listbox4.Bind (wx.EVT_CHOICE, self.Hero4)
        self.listbox5.Bind (wx.EVT_CHOICE, self.Hero5)
        self.listbox6.Bind (wx.EVT_CHOICE, self.Hero6)
        self.listbox7.Bind (wx.EVT_CHOICE, self.Hero7)
        self.listbox8.Bind (wx.EVT_CHOICE, self.Hero8)
        self.listbox9.Bind (wx.EVT_CHOICE, self.Hero9)
        self.listbox10.Bind (wx.EVT_CHOICE, self.Hero10)


        self.calculate = wx.Button(panel,-1,pos=(50,180),size=(100,40),label="calculate")
        self.calculate.Bind(wx.EVT_BUTTON,self.Calculateresult)

        wx.StaticText(panel, -1, "result:", (200, 190))
        self.result = wx.TextCtrl(panel,-1,'none',pos=(250,190),size=(300,30))

    def Calculateresult(self,event):
        list_predict[11] = -1
        list_predict[12] = int (random.randint (0, 3000000))
        data = [str (int (list_predict[0])), str (int (list_predict[1])), str (int (list_predict[2])),
                str (int (list_predict[3])), str (int (list_predict[4])), str (int (list_predict[5])),
                str (int (list_predict[6])), str (int (list_predict[7])), str (int (list_predict[8])),
                str (int (list_predict[9])), str (int (list_predict[10])), str (int (list_predict[11])),
                str (int (list_predict[12]))]
        fileHeader = ["", "radiant_1", "radiant_2", "radiant_3", "radiant_4", "radiant_5", "dire_1", "dire_2", "dire_3",
                      "dire_4", "dire_5", "radiant_win", "match_id"]
        csvFile = open ("instance.csv", "w")
        writer = csv.writer (csvFile)
        writer.writerow (fileHeader)
        writer.writerow (data)
        csvFile.close ()
        #data_for_predict = np_main.zeros((2,13))
        data_for_predict = iofunc.read_csv(os.getcwd() + "/instance.csv")
        #data_for_predict[1] = iofunc.read_csv (os.getcwd () + "/instance2.csv")
        #print(data_for_predict)
        #print(data_for_predict)
        data_for_predict.drop ('Unnamed: 0', axis=1, inplace=True)
        list_predict_feature = dota_predict._dataset_to_features (data_for_predict)
        #list_presict_fit = dota_predict.scaler.fit(list_predict)
        #print(list_predict_feature)
        x,y = list_predict_feature

        list_predict_scale = dota_predict.scaler.transform(x)
        result = dota_predict.model.predict(list_predict_scale)
        probality = dota_predict.model.predict_proba(list_predict_scale)
        #print(result)
        if(result[0]==1):
            self.result.Label="win" +" probality is "+str(probality[0][1])
        else:
            self.result.Label="lose"+" probality is "+str(probality[0][0])


    def Hero1(self,event):
        #print(self.listbox1.GetCurrentSelection())
        list_predict[1]=int(self.listbox1.GetCurrentSelection()+1)
        print(list_predict)

    def Hero2(self, event):
        list_predict[2] = int (self.listbox2.GetCurrentSelection ()+1)
        print (list_predict)

    def Hero3(self, event):
        list_predict[3] = int (self.listbox3.GetCurrentSelection ()+1)
        print (list_predict)

    def Hero4(self, event):
        list_predict[4] = int (self.listbox4.GetCurrentSelection ()+1)
        print (list_predict)

    def Hero5(self, event):
        list_predict[5] = int (self.listbox5.GetCurrentSelection ()+1)
        print (list_predict)

    def Hero6(self, event):
        list_predict[6] = int (self.listbox6.GetCurrentSelection ()+1)
        print (list_predict)

    def Hero7(self, event):
        list_predict[7] = int (self.listbox7.GetCurrentSelection ()+1)
        print (list_predict)

    def Hero8(self, event):
        list_predict[8] = int (self.listbox8.GetCurrentSelection ()+1)
        print (list_predict)

    def Hero9(self, event):
        list_predict[9] = int (self.listbox9.GetCurrentSelection ()+1)
        print (list_predict)

    def Hero10(self, event):
        list_predict[10] = int (self.listbox10.GetCurrentSelection())
        print (list_predict)

if __name__ == "__main__":
    list_predict = np_main.zeros(13)
    list_predict[0]=0
    #print(list_predict)

    #list_predict[1] = 1
    app = wx.App()
    frame = MyFrame()
    frame.Show()
    app.MainLoop()