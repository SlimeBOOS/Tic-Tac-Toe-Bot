import threading
import NeuralNet
import pyglet
import math
from pyglet.window import mouse
from pyglet.window import key
import time
import os


backgroundColor = [96, 96, 96]
textFont = 'Times New Roman'
textColor = [30, 28, 43]
settingsFile = "settings.dat"
playerDataDir = "Player Data"
playerDataName = "data"
batchDir = "Batches"
batchName = "batch"
settings = {}

class Scene:
    def __init__(self):
        self.width = 150
        self.height = 150
        self.buttons = list()
        self.boards = list()

    def draw(self):
        window.clear()
        for i in self.boards:
            i.drawBoard()

        for i in self.buttons:
            i.drawButton()
    
    def addButton(self, *args):
        self.buttons.append(Button(*args))

    def addBoard(self, x, y, width, height, lineWidth):
        self.boards.append(TTTBoard(x, y, width, height, lineWidth))

    def keyPress(self, key, modifiers):
        pass

    def keyRelease(self, key, modifiers):
        pass

class TTTBoard:
    def __init__(self, x, y, width, height, lineWidth):
        self.gapSizeX = (width-lineWidth*2)//3
        self.gapSizeY = (height-lineWidth*2)//3
        self.x = x
        self.y = y
        self.width = width - (width-lineWidth*2)%3
        self.height = height - (height-lineWidth*2)%3
        self.lineWidth = lineWidth
        self.lineColor = [45, 37, 66]
        self.OColor = [21, 92, 198]
        self.XColor = [219, 32, 32] 
        self.board = [0] * 9
        self.visible = True
        
        # If board[i] is 1 then its considered a "X"
        # If board[i] is -1 then its considered a "O"
        # If board[i] is 0 then its empty

    def setVisibility(self,value):
        self.visible = value

    def drawBoard(self):
        if not self.visible:
            return

        def drawSymbol(spacingX,spacingY,symbol):
            color = self.OColor
            pointers = []
            numPoints = 16
            if symbol == 1:
                pointers = [int(self.gapSizeX*0.3+spacingX), int(self.gapSizeY*0.5+spacingY), 
                            int(self.gapSizeX*0.1+spacingX), int(self.gapSizeY*0.3+spacingY), 
                            int(self.gapSizeX*0.3+spacingX), int(self.gapSizeY*0.1+spacingY), 
                            int(self.gapSizeX*0.5+spacingX), int(self.gapSizeY*0.3+spacingY), 
                            int(self.gapSizeX*0.7+spacingX), int(self.gapSizeY*0.1+spacingY), 
                            int(self.gapSizeX*0.9+spacingX), int(self.gapSizeY*0.3+spacingY), 
                            int(self.gapSizeX*0.7+spacingX), int(self.gapSizeY*0.5+spacingY), 
                            int(self.gapSizeX*0.9+spacingX), int(self.gapSizeY*0.7+spacingY), 
                            int(self.gapSizeX*0.7+spacingX), int(self.gapSizeY*0.9+spacingY), 
                            int(self.gapSizeX*0.5+spacingX), int(self.gapSizeY*0.7+spacingY), 
                            int(self.gapSizeX*0.3+spacingX), int(self.gapSizeY*0.9+spacingY),
                            int(self.gapSizeX*0.1+spacingX), int(self.gapSizeY*0.7+spacingY)]
                color = self.XColor
            elif symbol == -1:
                for i in range(numPoints):
                    angle = math.radians(i/numPoints * 360.0)
                    x = int((self.gapSizeX*0.4) * math.cos(angle) + (spacingX + self.gapSizeX*0.5))
                    y = int((self.gapSizeY*0.4) * math.sin(angle) + (spacingY + self.gapSizeY*0.5))
                    pointers += [x,y]                
            else:
                return
            drawPolygon(pointers, color)
            if symbol == -1:
                pointers = []
                for i in range(numPoints):
                    angle = math.radians(i/numPoints * 360.0)
                    x = int((self.gapSizeX*0.22) * math.cos(angle) + (spacingX + self.gapSizeX*0.5))
                    y = int((self.gapSizeY*0.22) * math.sin(angle) + (spacingY + self.gapSizeY*0.5))
                    pointers += [x,y]
                drawPolygon(pointers, backgroundColor)

        # Draw board
        for i in range(2):
            startX = self.x + self.gapSizeX*(i+1) + self.lineWidth * i
            startY = self.y
            width = self.lineWidth
            height = self.height
            pointers = getRectPoints(startX, startY, width, height)
            drawPolygon(pointers, self.lineColor)
        for i in range(2):
            startX = self.x
            startY = self.y + self.gapSizeY*(i+1) + self.lineWidth * i
            width = self.width
            height = self.lineWidth
            pointers = getRectPoints(startX, startY, width, height)
            drawPolygon(pointers, self.lineColor)
        # Draw X's and O's
        for i in range(len(self.board)):
            spacingX = self.x+(self.gapSizeX + self.lineWidth)*(i%3)
            spacingY = (self.y+self.height)-(self.gapSizeY + self.lineWidth)*(i//3) - self.gapSizeY
            drawSymbol(spacingX,spacingY,self.board[i])

    def reset(self):
        self.board = [0] * 9
    
    def save(self):
        if not os.path.isdir(playerDataDir):
            os.mkdir(playerDataDir)
        
        index = -1
        directory = os.listdir(playerDataDir)
        for i in range(len(directory)):
            if directory[i][:4] == playerDataName and int(directory[i][4]) > index:
                index = int(directory[i][4])
        
        
        fileName = playerDataName + str(index+1) + ".txt"
        with open(playerDataDir+"\\"+fileName, "w") as f:
            for i in range(8):
                f.write("{}:{};".format(i,self.board[i]))
            f.write("{}:{}".format(8,self.board[8]))

    def checkForWin(self):
        for i in [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(6,4,2)]:
            s = self.board[i[0]] + self.board[i[1]] + self.board[i[2]]
            if s in [3,-3]:
                return s/3
        if self.isFull():
            return 0
        
    def setSymbol(self,index,symbol):
        print(index,symbol)
        self.board[index] = symbol

    def isFull(self):
        for i in self.board:
            if i == 0:
                return False
        return True

class Button:
    def __init__(self, x=0, y=0, width=10, height=10, text='', function=None):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.text = text
        self.buttonColor = [54, 90, 147, 255]
        self.buttonFont = 'Times New Roman'
        self.buttonFontSize = 14
        self.buttonFontColor = [187, 201, 185, 255]
        self.function = function
        self.clicked = False
        self.visible = True
        
    def drawButton(self):
        if not self.visible:
            return
        pointers = getRectPoints(self.x,self.y,self.width,self.height)
        labelX = self.x + self.width//2
        labelY = self.y + self.height//2
        label = pyglet.text.Label(self.text,font_name=self.buttonFont,font_size=self.buttonFontSize,x=labelX,y=labelY,anchor_x='center',anchor_y='center',color=self.buttonFontColor)
        color = self.buttonColor
        if (self.clicked):
            color = list(int(0.8*i) for i in color)

        drawPolygon(pointers, color)
        label.draw()

    def setVisibility(self,value):
        self.visible = value

    def clickDetection(self, x, y, button):
        if self.visible and (self.x < x < self.x + self.width) and (self.y < y < self.y + self.height):
            self.clicked = True

    def resetClicked(self):
        if (self.clicked):
            if self.function != None:
                self.function()
            self.clicked = False
    
def changeScene(name):
    global currentScene
    currentScene = scenes[name]
    if type(currentScene) == TrainingScene:
        currentScene.updateBatch()
    window.set_size(currentScene.width,currentScene.height)

def getRectPoints(x,y,width,height):
    return [x,y,x+width,y,x+width,y+height,x,y+height]

def drawPolygon(vertices, color):
    colorCode = 'c3B'
    if len(color) == 4:
        colorCode = 'c4B'
    
    if len(vertices) % 2 == 0:
        pyglet.gl.glEnable(pyglet.gl.GL_BLEND)
        pyglet.gl.glBlendFunc(pyglet.gl.GL_SRC_ALPHA, pyglet.gl.GL_ONE_MINUS_SRC_ALPHA)
        pyglet.graphics.draw(len(vertices)//2, pyglet.gl.GL_POLYGON, ('v2i', vertices), (colorCode, color * (len(vertices)//2)))
        pyglet.gl.glDisable(pyglet.gl.GL_BLEND)
    else:
        raise ValueError("The length of vertices({}) must be divisible by 2!".format(len(vertices)))

def drawLabel(text,labelX,labelY,fontSize,fontColor=textColor,anchorX='center',anchorY='center'):
    if len(fontColor) == 3:
        fontColor.append(255)
    label = pyglet.text.Label(text,font_name=textFont,font_size=fontSize,x=labelX,y=labelY,color=fontColor,anchor_x=anchorX,anchor_y=anchorY)
    label.draw()

def setSetting(settingsDict, key, value, text):
    settingsDict[key] = [value, text]

def saveSettings(settingsDict,fileName):
    with open(fileName,"w") as f:
        for i in settingsDict.keys():
            f.write("{} {} {}\n".format(i, settingsDict[i][0], settingsDict[i][1]))

def loadSettings(settingsDict,fileName):
    settingsDict.clear()
    setSetting(settingsDict, "saveGame", True, "Save games")
    setSetting(settingsDict, "mutationRate", 0.01, "Mutation rate")
    setSetting(settingsDict, "poolSize", 10, "Pool size") 
    setSetting(settingsDict, "batchIndex", 0, "Batch index") 
    setSetting(settingsDict, "crossoverChance", 0.75, "Crossover chance") 

    if (os.path.isfile(fileName)):
        with open(fileName,"r") as f:
            data = f.read()
            for i in data[:-1].split("\n"):
                if (len(i) > 1):
                    spaceIndex = i.find(" ")
                    key = i[:spaceIndex]
                    value = type(settingsDict[key][0])(i[spaceIndex+1:i.find(" ",spaceIndex+1)])
                    text = i[i.find(" ",spaceIndex+1)+1:]
                    
                    setSetting(settingsDict, key,value,text)

loadSettings(settings, settingsFile)

class MenuScene(Scene):
    def __init__(self):
        super().__init__()
        self.width = 200
        self.height = 180
        self.addButton(5, self.height-30, 190,25, "Play a game",lambda: changeScene("PlayGame"))
        self.addButton(5, self.height-60, 190,25, "Train on player data", lambda: changeScene("TrainOnPlayer"))
        self.addButton(5, self.height-90, 190,25, "Train on neural networks", lambda: changeScene("TrainOnNeural"))
        self.addButton(5, self.height-120, 190,25, "Edit settings",lambda: changeScene("Settings"))

    def draw(self):
        super().draw()
        drawLabel("Made by SlimeBOOS", 5, 5, 12, anchorX="left", anchorY="bottom")

class PlayGameScene(Scene):
    def __init__(self):
        super().__init__()
        self.width = 350
        self.height = 300
        self.turn = 1
        self.state = 0
        self.addBoard(self.width//2-125,10,250,250,10)
        self.addButton(5, self.height-30, 50, 25, "Back",lambda: changeScene("Menu"))
        self.addButton(5, self.height-60, 50, 25, "Reset",self.reset)
        
        spacingX = lambda i: self.boards[0].x+(self.boards[0].gapSizeX + self.boards[0].lineWidth)*(i%3)
        spacingY = lambda i: (self.boards[0].y+self.boards[0].height)-(self.boards[0].gapSizeY + self.boards[0].lineWidth)*(i//3) - self.boards[0].gapSizeY

        self.addButton(spacingX(0),spacingY(0),self.boards[0].gapSizeX,self.boards[0].gapSizeY,"", lambda: self.setSymbol(0))
        self.addButton(spacingX(1),spacingY(1),self.boards[0].gapSizeX,self.boards[0].gapSizeY,"", lambda: self.setSymbol(1))
        self.addButton(spacingX(2),spacingY(2),self.boards[0].gapSizeX,self.boards[0].gapSizeY,"", lambda: self.setSymbol(2))
        self.addButton(spacingX(3),spacingY(3),self.boards[0].gapSizeX,self.boards[0].gapSizeY,"", lambda: self.setSymbol(3))
        self.addButton(spacingX(4),spacingY(4),self.boards[0].gapSizeX,self.boards[0].gapSizeY,"", lambda: self.setSymbol(4))
        self.addButton(spacingX(5),spacingY(5),self.boards[0].gapSizeX,self.boards[0].gapSizeY,"", lambda: self.setSymbol(5))
        self.addButton(spacingX(6),spacingY(6),self.boards[0].gapSizeX,self.boards[0].gapSizeY,"", lambda: self.setSymbol(6))
        self.addButton(spacingX(7),spacingY(7),self.boards[0].gapSizeX,self.boards[0].gapSizeY,"", lambda: self.setSymbol(7))
        self.addButton(spacingX(8),spacingY(8),self.boards[0].gapSizeX,self.boards[0].gapSizeY,"", lambda: self.setSymbol(8))

        for i in range(9):
            self.buttons[i+2].buttonColor[3] = 0
    
    def reset(self):
        self.boards[0].reset()
        self.turn = 1
        
    def setSymbol(self,index):
        if (self.boards[0].board[index] == 0):
            self.boards[0].board[index] = self.turn
            self.turn *= -1

    def draw(self):
        super().draw()
        color = self.boards[0].OColor
        symbol = "O"
        if (self.turn == 1):
            color = self.boards[0].XColor
            symbol = "X"

        text = symbol+"'s turn"
    
        result = self.boards[0].checkForWin()
        if (result == 1):
            text = "X WON!"
            color = self.boards[0].XColor
            
            if settings["saveGame"][0]:
                self.boards[0].save()
            self.reset()
        elif (result == -1):
            text = "O WON!"
            color = self.boards[0].OColor
            
            if settings["saveGame"][0]:
                self.boards[0].save()
            self.reset()
        elif (result == 0):
            text = "Draw!"
            color = textColor

            if settings["saveGame"][0]:
                self.boards[0].save()
            self.reset()
        
        drawLabel(text,self.width//2,self.height-15,20,color)

class SettingsScene(Scene):
    def __init__(self):
        super().__init__()
        self.width = 300
        self.height = 250
        self.addButton(5, self.height-30, 50, 25, "Back",lambda: changeScene("Menu"))
        self.textSize = 14
        self.increment = 1.0
        self.incrementPlusKey = key.E
        self.incrementMinusKey = key.Q

        index = 0
        for i in settings:
            settingType = type(settings[i][0])
            posY = self.height - 60 - (index * 25)
            index += 1
            label = pyglet.text.Label("{}: {}".format(settings[i][1],settings[i][0]),font_name=textFont,font_size=self.textSize)
            height = label.content_height
            if settingType == float or settingType == int:
                self.addButton(5,posY,30,height,"-", self.buttonHandler)
                self.addButton(40,posY,30,height,"+", self.buttonHandler)
            elif settingType == bool:
                self.addButton(5,posY,50,height,"FLIP", self.buttonHandler)

        self.textOffsetX = self.buttons[0].x + self.buttons[0].width
        for i in self.buttons[1:]:
            if (i.x + i.width > self.textOffsetX):
                self.textOffsetX = i.x + i.width

    def buttonHandler(self):
        for i in self.buttons:
            if i.clicked == True:
                index = (self.height - i.y - 60)//25
                key = list(settings.keys())[index]
                settingType = type(settings[key][0])
                if settingType == float or settingType == int:
                    if i.text == "+":
                        settings[key][0] += self.increment
                    else:
                        settings[key][0] -= self.increment
                    if settingType == float:
                        settings[key][0] = round(settings[key][0],8)
                    else:
                        settings[key][0] = int(settings[key][0])
                elif settingType == bool:
                    settings[key][0] = not settings[key][0]
                break

    def draw(self):
        super().draw()
        
        drawLabel("Increment size: {}".format(self.increment),self.width//2,self.height - 12,self.textSize)
        
        plusSymbol = key.symbol_string(self.incrementPlusKey)
        minusSymbol = key.symbol_string(self.incrementMinusKey)
        drawLabel("{} - /10 Increment   {} - x10 Increment".format(plusSymbol, minusSymbol),self.width//2,5,self.textSize-2,anchorY="bottom")
    
        index = 0
        for i in settings:
            offsetY = (index * 25)
            index += 1
            drawLabel("{}: {}".format(settings[i][1],settings[i][0]),self.textOffsetX+10,self.height-60-offsetY,self.textSize,anchorX='left',anchorY='bottom')

    def keyRelease(self, keyCode, modifiers):
        super().keyRelease(keyCode, modifiers) 
        if (keyCode == self.incrementMinusKey and self.increment > 1E-5):
            self.increment /= 10
        elif (keyCode == self.incrementPlusKey and self.increment < 1E5):
            self.increment *= 10

class TrainingScene(Scene):
    def __init__(self, dataType):
        super().__init__()
        saveSettings(settings, settingsFile)

        self.width = 200
        self.height = 200
        self.addButton(5, self.height-30, 50, 25, "Back", self.Exit)
        self.addButton(60, self.height-30, 65, 25, "Restart", self.restart)
        self.updateBatch()
        self.pool = []
        self.textSize = 14
        self.updateBatch()
        self.newPool()

        loadSettings(settings, self.batchDir+"\\"+settingsFile)
        
    def updateBatch(self):
        if not os.path.isdir(batchDir):
            os.mkdir(batchDir)

        if not os.path.isdir(batchDir+"/"+batchName + " " + str(settings["batchIndex"][0])):
            os.mkdir(batchDir+"/"+batchName + " " + str(settings["batchIndex"][0]))

        self.batchDir = batchDir+"/"+batchName + " " + str(settings["batchIndex"][0])

    def restart(self):
        self.pool = []
        for _ in range(settings["poolSize"][0]):
            self.pool.append(NeuralNet.NEAT(9, 9))

        loadSettings(settings, settingsFile)
        saveSettings(settings, self.batchDir+"\\"+settingsFile)

    def Exit(self):
        saveSettings(settings, self.batchDir+"\\"+settingsFile)
        loadSettings(settings, settingsFile)
        changeScene("Menu")

    def fitnessFunction(self,neuralNet):
        pass

    def doEpoch(self):
        pass

    def newPool(self):
        oldPool = self.pool.copy()
        self.pool = []
        if len(oldPool) == 0:
            for _ in range(settings["poolSize"][0]):
                self.pool.append(NeuralNet.NEAT(9, 9))
        else:
            pass

    def draw(self):
        super().draw()

        skipSettings = {"saveGame","batchIndex"}
        index = 0
        for key, value in settings.items():
            if key in skipSettings:
                continue
            drawLabel("{}: {}".format(value[1], value[0]),10,self.height-45-(index*25),self.textSize,anchorX='left')
            index += 1

window = pyglet.window.Window(resizable=False) 
pyglet.gl.glClearColor(*(list(i/255 for i in backgroundColor) + [1]))

scenes = {"Menu": MenuScene(),"PlayGame":PlayGameScene(),"Settings":SettingsScene(),"TrainOnPlayer": TrainingScene("Player"),"TrainOnNeural": TrainingScene("Neural")}
changeScene("Menu")

@window.event
def on_mouse_press(x, y, button, modifiers):
    for i in currentScene.buttons:
        i.clickDetection(x, y, button)

@window.event
def on_key_press(key, modifiers):
    currentScene.keyPress(key, modifiers)

@window.event
def on_key_release(key, modifiers):
    currentScene.keyRelease(key, modifiers)

@window.event
def on_mouse_release(x, y, button, modifiers):
    for i in currentScene.buttons:
        i.resetClicked()

@window.event
def on_draw():
    currentScene.draw()

pyglet.app.run()

saveSettings(settings, settingsFile)