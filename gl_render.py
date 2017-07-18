#!/usr/bin/env python

from OpenGL.GLUT import *
from OpenGL.GLU import *
from OpenGL.GL import *
from OpenGL.GLX import *
import sys
import numpy as np
from ctypes import *
import random
from PIL import Image
import PIL

name = 'ball_glut'


class Quad:
  def __init__(self):
    self.vao = None
    self.vbo = None
    self.ibo = None
    self.iddata = np.array([[0,1,2], [2,3,0]], dtype=np.int32)
    self.vxdata = np.array([[-1.0, -1.0, 0.0,   0.0, 0.0, 0.0, 1.0],
                            [ 1.0, -1.0, 0.0,   1.0, 0.0, 0.0, 1.0],
                            [ 1.0,  1.0, 0.0,   1.0, 0.0, 1.0, 1.0],
                            [-1.0,  1.0, 0.0,   0.0, 0.0, 1.0, 1.0]], dtype=np.float32)
    #self.vxdata = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    #                        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    #                        [1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0],
    #                        [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]], dtype=np.float32)
    #self.coldata = np.array([[1,0,0,1], [0,1,0,1], [1,1,1,0], [0,0,1,1]], dtype=np.float32)
  def init(self):
    self.vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
    print(self.vxdata.flatten())
    glBufferData(GL_ARRAY_BUFFER, self.vxdata, GL_STATIC_DRAW)
    
    self.vao = glGenVertexArrays(1)
    
    self.ibo = glGenBuffers(1)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ibo)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, self.iddata, GL_STATIC_DRAW)
    
  def bind(self):
    glBindVertexArray(self.vao)
    glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 28, None)
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 28, c_void_p(12))
    glEnableVertexAttribArray(0)
    glEnableVertexAttribArray(1)
    
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ibo)
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)
    
    glDisableVertexAttribArray(1)
    glDisableVertexAttribArray(0)
    
  def destroy(self):
    if self.vbo: 
      glDeleteBuffers(1, [self.vbo])
      self.vbo = None
    if self.ibo: 
      glDeleteBuffers(1, [self.ibo])
      self.ibo = None
    if self.vao: 
      glDeleteVertexArrays(1, [self.vao])
      self.vao = None    
      
class shader_program:
  def __init__(self, vertex_code, fragment_code):
    self.vertex_code = vertex_code
    self.fragment_code = fragment_code
    self.vertex_program = None
    self.fragment_program = None
    
    self.shader = None
  
  def init(self):
    try:
      self.vertex_program = glCreateShader(GL_VERTEX_SHADER)
      self.fragment_program = glCreateShader(GL_FRAGMENT_SHADER)
      
      glShaderSource(self.vertex_program, self.vertex_code)
      glCompileShader(self.vertex_program)
      result = glGetShaderiv(self.vertex_program, GL_COMPILE_STATUS)
      if result != GL_TRUE:
        raise RuntimeError(glGetShaderInfoLog(self.vertex_program))
        
      glShaderSource(self.fragment_program, self.fragment_code)
      glCompileShader(self.fragment_program)
      result = glGetShaderiv(self.fragment_program, GL_COMPILE_STATUS)
      if result != GL_TRUE:
        raise RuntimeError(glGetShaderInfoLog(self.fragment_program))
      
      self.shader = glCreateProgram()
      glAttachShader(self.shader, self.vertex_program)
      glAttachShader(self.shader, self.fragment_program)
      glLinkProgram(self.shader)
      
      result = glGetProgramiv(self.shader, GL_LINK_STATUS)
      if result != GL_TRUE:
        raise RuntimeError(glGetProgramInfoLog(self.shader))
      
      glDetachShader(self.shader, self.vertex_program)
      glDetachShader(self.shader, self.fragment_program)
    except:
      self.destroy()
      raise
  def destroy(self):
    if self.vertex_program: 
      glDeleteShader(self.vertex_program)
      self.vertex_program = None
    if self.fragment_program: 
      glDeleteShader(self.fragment_program)
      self.fragment_program = None
    
    if self.shader: 
      glDeleteProgram(self.shader)
      self.shader = None
    
  def bind(self):
    glUseProgram(self.shader)
  @staticmethod
  def unbind():
    glUseProgram(0)
    
  def draw(self, callable):
    self.bind()
    callable()
    self.unbind()
    

VERTEX_SOURCE = '''
#version 330 core
layout(location = 0) in vec3 position;
layout(location = 1) in vec4 color;

out vec4 col;

void main(){
  gl_Position = vec4(position.x, -position.y, 0.0f, 1.0f);
  col = color;
}
'''

FRAGMENT_SOURCE = '''
#version 330 core
layout(location = 0) out vec4 color;
layout(location = 1) out float alpha;

in vec4 col;

void main(){
  color = vec4(col.xyz * 2, 0.5f);
  alpha = 0.5f;
}
'''

class target_texture:
  RGBA = 0x4
  R = 0x1
  RED = 0x1
  MONO = 0x1
  RGB = 0x3
  def __init__(self, size = (512, 512), channels = 0x4):
    self.size = size
    self.channels = channels
    self._texture = None
    self.data = []
  
  def init(self):
    self._texture = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, self._texture)
    
    glTexImage2D(GL_TEXTURE_2D, 0, self.__gl_channels_internal(), self.size[0], self.size[1], 0, self.__gl_channels_type(), GL_FLOAT, None)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
  
  @staticmethod
  def unbind():
    glBindTexture(GL_TEXTURE_2D, 0)
  
  def read(self):
    self.data = glGetTexImage(GL_TEXTURE_2D, 0, self.__gl_channels_type(), GL_FLOAT, None)
  
  def destroy(self):
    self.unbind()
    if self._texture:
      glDeleteFramebuffers(1, [self._texture])
      self._texture = None
  
  def __gl_channels_internal(self):
    if self.channels == 1:
      return GL_R32F
    if self.channels == 3:
      return GL_RGB
    if self.channels == 4:
      return GL_RGBA
  def __gl_channels_type(self):
    if self.channels == 1:
      return GL_RED
    if self.channels == 3:
      return GL_RGB
    if self.channels == 4:
      return GL_RGBA

class framebuffer:
  def __init__(self, size = (512, 512)):
    self.fbo = None
    self.data = []
    self.size = size
    
    #self._renderTexture = None
    #self._renderAlpha = None
    self._depthBuffer = None
    self._drawBuffers = {}
    
  def attach(self, target, texture):
    assert(isinstance(texture, target_texture))
    self._drawBuffers[target] = texture
    #glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + target, texture._texture, 0);
    
  def init(self):
    self.fbo = glGenFramebuffers(1)
    glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)
    
    #if self.__rgb:
    #self._renderTexture = glGenTextures(1)
    #glBindTexture(GL_TEXTURE_2D, self._renderTexture)
    
    #glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, self.size[0], self.size[1], 0, GL_RGBA, GL_FLOAT, None)
    #glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    #glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    #glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, self._renderTexture, 0);
    
    ##if self.__alpha:
    #self._renderAlpha = glGenTextures(1)
    #glBindTexture(GL_TEXTURE_2D, self._renderAlpha)
    
    #glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, self.size[0], self.size[1], 0, GL_RED, GL_FLOAT, None)
    #glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    #glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    #glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, self._renderAlpha, 0);
  
    #self._depthBuffer = glGenRenderbuffers(1)
    #glBindRenderbuffer(GL_RENDERBUFFER, self._depthBuffer)
    #glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, self.size[0], self.size[1])
    #glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, self._depthBuffer)

    #glDrawBuffers([GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1]);
    
  def bind(self):
    glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)
    
    for k, v in self._drawBuffers.items():
      glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + k, v._texture, 0);
    glDrawBuffers([x + GL_COLOR_ATTACHMENT0 for x in self._drawBuffers.keys()]);

    glViewport(0,0, self.size[0], self.size[1])
  
  @staticmethod
  def unbind():
    glBindFramebuffer(GL_FRAMEBUFFER, 0)
  
  def draw(self, callable):
    self.bind()
    callable()
    self.unbind()
    
  def render(self, scene):
    def __render():
      #glClearColor(0.0,0.0,0.0,0.0)
      #glClear(GL_COLOR_BUFFER_BIT)
      #glEnable(GL_BLEND)
      #glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
      #glBlendFuncSeparate(GL_ONE, GL_ZERO,
      #              GL_ONE, GL_ZERO)
      #glBlendEquationSeparate(GL_MAX, GL_MIN)
      ##glDisable( GL_ALPHA_TEST )
      #glEnable( GL_BLEND );
      #glBlendFunc(GL_ONE, GL_ZERO)
      #glBlendFuncSeparate( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, 
      #                     GL_SRC_ALPHA, GL_ONE );
      scene()
      for v in self._drawBuffers.values():
        v.read()
      #glBindTexture(GL_TEXTURE_2D, self._renderTexture)
      #self.data = glReadPixels(0, 0, self.size[0], self.size[1], GL_RGBA, GL_FLOAT, None)
      #self.data = glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_FLOAT, None)
      #glBindTexture(GL_TEXTURE_2D, self._renderAlpha)
      #self.data[:,:,3] = glGetTexImage(GL_TEXTURE_2D, 0, GL_RED, GL_FLOAT, None)
      
      #print(self.data)
    self.draw(__render)
    
  def save_pic(self, target, filename):
    arr = self._drawBuffers[target].data[::-1,:,:]
    #print(arr)
    im = Image.fromarray(np.uint8(arr*255))
    im.save(filename)
    
  def destroy(self):
    self.unbind()
    if self.fbo:
      glDeleteFramebuffers(1, [self.fbo])
      self.fbo = None
    if self._depthBuffer:
      glDeleteRenderbuffers(1, [self._depthBuffer])
      self._depthBuffer = None
    #if self._renderTexture:
      #glDeleteTextures([self._renderTexture])
      #self._renderTexture = None
    #if self._renderAlpha:
      #glDeleteTextures([self._renderAlpha])
      #self._renderAlpha = None
    

class gl_filter:
  def __init__(self, size = (512, 512)):
    self._inputs = []
    self._names = {}
    self._outputs = []
    self.fb = framebuffer(size)
    
  def init(self):
    max_textures = glGetInteger(GL_MAX_COMBINED_TEXTURE_IMAGE_UNITS)
    if len(self._inputs) + len(self._outputs) > max_textures:
      raise RuntimeError('Not capable')


def main():
    #color = [1.0,0.,0.,1.]
    #glMaterialfv(GL_FRONT,GL_DIFFUSE,color)
    #glutSolidSphere(2,20,20)
    
    
    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
    glutInitWindowSize(400,400)
    glutCreateWindow(name)
    
    #print()
    
    glClearColor(0.,0.,0.,1.)
    #glShadeModel(GL_SMOOTH)
    #glEnable(GL_CULL_FACE)
    glEnable(GL_DEPTH_TEST)
    #glCullFace(GL_FRONT_AND_BACK)
    glDisable(GL_LIGHTING)
    #lightZeroPosition = [10.,4.,10.,1.]
    #lightZeroColor = [0.8,1.0,0.8,1.0] #green tinged
    #glLightfv(GL_LIGHT0, GL_POSITION, lightZeroPosition)
    #glLightfv(GL_LIGHT0, GL_DIFFUSE, lightZeroColor)
    #glLightf(GL_LIGHT0, GL_CONSTANT_ATTENUATION, 0.1)
    #glLightf(GL_LIGHT0, GL_LINEAR_ATTENUATION, 0.05)
    #glEnable
    q = Quad()
    fb = framebuffer()
    sh = shader_program(VERTEX_SOURCE, FRAGMENT_SOURCE)
    target = target_texture()
    
    def display():
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
        #glClearColor(random.random(),0.,0.,1.)
        glPushMatrix()
        #glRotatef(random.random() * 6.28, 0, 0, 1)
        #glColor3f(1,1,1)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        sh.bind()
        q.bind()
        glPopMatrix()
        glutSwapBuffers()
        return
    glutDisplayFunc(display)
    
    q.init()
    sh.init()
    fb.init()
    target.init()
    
    fb.attach(0, target)
    
    fb.render(lambda: sh.draw(q.bind))
    fb.save_pic(0, 'test_img.png')
    
    
    #glMatrixMode(GL_PROJECTION)
    #glLoadIdentity()
    #glOrtho(0, 1, 0, 1, 0, 10)
    #gluPerspective(40.,1.,1.,40.)
    #glMatrixMode(GL_MODELVIEW)
    #gluLookAt(0, 0, 5, 0, 0, 0, 0, 1, 0)
    #glLoadIdentity()
    #glPushMatrix()
    #glutMainLoop()
    #glPopMatrix()
    
    fb.destroy()
    target.destroy()
    sh.destroy()
    q.destroy()
    return



if __name__ == '__main__':
  main()
  print('ended')
