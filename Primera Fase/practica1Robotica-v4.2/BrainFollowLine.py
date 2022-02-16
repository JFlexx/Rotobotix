###################### Autores ###########################
# --> Jason Felipe Vaz
##########################################################


from pyrobot.brain import Brain

import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from pyrobot.tools.followLineTools import findLineDeviation
import math

class BrainFollowLine(Brain):

  ############################### Variables declaradas ########################
  esquivaObs= False # Nos indicara cuando hay que empezar a esquivar dado un obstaculo
  pierdeLinea= False # Servira para activar el movimiento hacia atras
  reposicion= False # Variable que reposicionara al robot en linea al esquivar obstaculo
  ###########################################################################
  NO_FORWARD = 0
  SLOW_FORWARD = 0.1
  MED_FORWARD = 0.5
  FULL_FORWARD = 1.0
  
  NO_TURN = 0
  MED_LEFT = 0.5
  HARD_LEFT = 1.0
  MED_RIGHT = -0.5
  HARD_RIGHT = -1.0

  #NO_ERROR = 0#Esta variable no se usará

  def setup(self):
    self.image_sub = rospy.Subscriber("/image",Image,self.callback)
    self.bridge = CvBridge()

  def callback(self,data):
    self.rosImage = data

  def destroy(self):
    cv2.destroyAllWindows()

  def step(self):
    # take the last image received from the camera and convert it into
    # opencv format
    try:
      cv_image = self.bridge.imgmsg_to_cv2(self.rosImage, "bgr8")
    except CvBridgeError as e:
      print(e)

    # display the robot's camera's image using opencv
    cv2.imshow("Stage Camera Image", cv_image)
    cv2.waitKey(1)

    # write the image to a file, for debugging etc.
    # cv2.imwrite("debug-capture.png", cv_image)

    # convert the image into grayscale
    imageGray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

    # determine the robot's deviation from the line.
    foundLine,error = findLineDeviation(imageGray)
    # print("findLineDeviation returned ",foundLine,error)

#     # display a debug image using opencv
#     middleRowIndex = cv_image.shape[1]//2
#     centerColumnIndex = cv_image.shape[0]//2
#     if (foundLine):
#       cv2.rectangle(cv_image,
#                     (int(error*middleRowIndex)+middleRowIndex-5,
#                      centerColumnIndex-5),
#                     (int(error*middleRowIndex)+middleRowIndex+5,
#                      centerColumnIndex+5),
#                     (0,255,0),
#                     3)
#     cv2.imshow("Debug findLineDeviation", cv_image)
#     cv2.waitKey(1)

  # A trivial on-off controller
  ################### Variables de utilidad ###############################
    # Con los 8 sensores obtenemos las distancias hacia los objetos
    front = min([s.distance() for s in self.robot.range["front"]])
    left = min([s.distance() for s in self.robot.range["left-front"]])
    right = min([s.distance() for s in self.robot.range["right-front"]])
    leftEdge= min([s.distance() for s in self.robot.range["front-left"]]) 
    rightEdge= min([s.distance() for s in self.robot.range["front-right"]])


  ############### Prints que ayudan a entender el avance del robot ########
    # print("Esquivaobstaculos", self.esquivaObs)
    # print("FoundLine", foundLine)
    # print("bloqueo general")
    # print("Error",error)
    # print("-------------------------------------------")

  ###################### Estado: Esquivar obsaculo ###############################
    #--> En caso de que exista algún obstáculo
    if ( front < 0.5 or leftEdge < 0.5 or rightEdge < 0.6):
      # Hay que esquivarlo, activamos el estado para esquivar
      self.esquivaObs=True
      # Realizo giro hacia la izquierda sin avanzar
      self.move(0, self.HARD_LEFT)

    #--> Entramos en el estado de "esquivar obstáculo"
    if (self.esquivaObs):

      #--> Gira a la izquiera al principio del obstaculo(1º movimiento)
      if (rightEdge < 0.4 ):
        self.move(self.NO_FORWARD, self.MED_LEFT)# Gira sin avanzar, para evitar choque

      #--> Avanzo a medida que giro a la derecha(2º movimiento)
      elif (right < 0.5):
        self.esquivaObs = not foundLine# Si esto se pone a false
        if (self.esquivaObs==False): self.reposicion=True# Esto activara el modo Reposicion
        self.move(self.MED_FORWARD, self.MED_RIGHT)

      #--> Servira para no girar demasiado, o desviarnos en el giro(3º movimiento) en el borde de los obstaculos
      elif (right> 0.5 and right < 1):
        self.move(self.SLOW_FORWARD, self.HARD_RIGHT)

  ############################ Estado: Seguir linea #####################################
    #--> Sigue la línea usando función
    elif (foundLine):

      #--> Esto ser un movimiento de emergencia en caso de perder la linea
      self.pierdeLinea= foundLine# Guardando el estado de "foundLine" 

      #--> Modo Reposicion del robot
      if (self.reposicion== True):# Solo se realizará 1 vez despues de salir del obstaculo
        self.move(self.MED_FORWARD, self.HARD_LEFT)# Evita de tal forma que el robot quede perdendicular a la linea
        self.reposicion= False
        return

      #--> Aplico función para seguir la linea
      turnVelocity= -1*error;#TV
      forwardVelocity= max(0,1-math.fabs(turnVelocity*1.5))# FV
      self.move(forwardVelocity, turnVelocity)# Cuando mayor sea el giro, mas lento avanzara el robot.
    
    #--> Se activa el movimiento hacia atras hasta que encuentre la linea
    elif (not foundLine and self.pierdeLinea):
      self.move(-self.MED_FORWARD,0)

    #--> En caso de que empiece el robot sin estar en ninguna linea, por lo que buscara la linea
    else:
      self.move(self.FULL_FORWARD,0)# Simple movimiento hacia delante(Quizas no es lo mas adecuado)

##################################################### Búsqueda espiral #######################################################################
  #  pi = 3.14159265358979323846
  #   n=1500
  #   r=3500
  #   t=0
  #   while(not foundLine and t<=1):
  #     if(foundLine):
  #       print("foundLine")
  #       break
  #     print("que cojones", foundLine)
  #     print("fuck")
  #     print("Error", error)
  #     x=r*t*math.sin(t*2*pi*n)
  #     y=r*t*math.cos(t*2*pi*n)
  #     t += 1.0/(360.0*n)
  #     self.move(x,y)
  #Esta parte del código estaba en la función de estep y entraba en un bucle cuando al principio del circuito no encontraba la linea
##################################################### Búsqueda espiral #######################################################################


def INIT(engine):
  assert (engine.robot.requires("range-sensor") and
	  engine.robot.requires("continuous-movement"))

  return BrainFollowLine('BrainFollowLine', engine)
