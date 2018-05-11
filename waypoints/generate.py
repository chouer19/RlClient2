import math

def writeFile(name,ax,ay,yaw,fromX,toX,fromY,toY):
    with open(name,'w') as f:
        if yaw == 90 or yaw == -90:
            ax = 0
        elif yaw == -180 or yaw == 180:
            ax *= -1
        if yaw == -90:
            ay *= -1
        elif yaw % 180 == 0:
            ay = 0
        XX = fromX
        YY = fromY
        while XX <= max(toX,fromX) and XX >= min(toX,fromX) and YY <= max(toY,fromY) and YY >= min(toY,fromY):
            f.write(str(XX)+',' +str(YY)+',' +str(yaw)+'\n')
            XX +=  ax
            XX = round(XX,2)
            YY +=  ay
            YY = round(YY,2)
            print(XX,YY,yaw)
        #f.write(str(toX)+',' +str(toY)+',' +str(yaw)+'\n')

ax=0.1
ay=0.1

name='1.txt'
yaw=180
fromX=271
toX=90.5+12
fromY=131.5
toY=131.5
writeFile(name,ax,ay,yaw,fromX,toX,fromY,toY)

name='2.txt'
yaw=-90
fromX=90.5
toX=90.5
fromY=131.5-12
toY=0+12
writeFile(name,ax,ay,yaw,fromX,toX,fromY,toY)

name='3.txt'
yaw=180
fromX=90.5-12
toX=12
fromY=0
toY=0
writeFile(name,ax,ay,yaw,fromX,toX,fromY,toY)

name='4.txt'
yaw=90
fromX=0
toX=0
fromY=0+12
toY=329-12
writeFile(name,ax,ay,yaw,fromX,toX,fromY,toY)

name='5.txt'
yaw=0
fromX=12
toX=90.5-12
fromY=329
toY=329
writeFile(name,ax,ay,yaw,fromX,toX,fromY,toY)

name='6.txt'
yaw=-90
fromX=90.5
toX=90.5
fromY=329-12
toY=197.5+12
writeFile(name,ax,ay,yaw,fromX,toX,fromY,toY)

name='7.txt'
yaw=0
fromX=90.5+12
toX=337-12
fromY=197.5
toY=196.5
writeFile(name,ax,ay,yaw,fromX,toX,fromY,toY)

name='8.txt'
yaw=90
fromX=337
toX=337
fromY=197.5+12
toY=329-12
writeFile(name,ax,ay,yaw,fromX,toX,fromY,toY)

name='9.txt'
yaw=0
fromX=337+12
toX=394.5-12
fromY=329
toY=329
writeFile(name,ax,ay,yaw,fromX,toX,fromY,toY)

name='10.txt'
yaw=-90
fromX=394.5
toX=394.5
fromY=329-12
toY=12
writeFile(name,ax,ay,yaw,fromX,toX,fromY,toY)

name='11.txt'
yaw=-180
fromX=394.5-12
toX=156+12
fromY=0
toY=0
writeFile(name,ax,ay,yaw,fromX,toX,fromY,toY)

name='12.txt'
yaw=90
fromX=156
toX=156
fromY=0+12
toY=57.5-12
writeFile(name,ax,ay,yaw,fromX,toX,fromY,toY)

name='13.txt'
yaw=0
fromX=156+12
toX=337-12
fromY=57.5
toY=57.5
writeFile(name,ax,ay,yaw,fromX,toX,fromY,toY)

name='14.txt'
yaw=90
fromX=337
toX=337
fromY=57.5+12
toY=131.5-12
writeFile(name,ax,ay,yaw,fromX,toX,fromY,toY)

name='15.txt'
yaw=-180
fromX=337-12
toX=271.1
fromY=131.5
toY=131.5
writeFile(name,ax,ay,yaw,fromX,toX,fromY,toY)
