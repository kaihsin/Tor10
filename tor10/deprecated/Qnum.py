import torch, copy
import numpy as np

class Qnum_list:
	"""List of qauntum numbers"""
	def __init__(self,qnstring):
		self.__qnstring=qnstring


class Qnum_base:
	"""Quantum number base class"""
	def __init__(self):
		self._q=0
	@property
	def q(self):
		return self._q
	@q.setter
	def q(self,q):
		self._q=q

class U1(Qnum_base):
	"""docstring for U1"""

	def __init__(self, q=0):
		super().__init__()
		self._q=q


	def __neg__(self):
		self._q=-self._q
	def __add__(self, u):
		if isinstance(u,U1) :
			return U1(self.q+u.q)
		else:
			raise TypeError("Cannot add quantum numbers of differe types!")


class Zn(Qnum_base):
	"""docstring for Zn"""
	def __init__(self, n=2,q=0):
		super().__init__()
		self._mod=n
		self._q=q % self._mod
	@property
	def q(self):
		return self._q
	@q.setter
	def q(self,q):
		self._q=q % self._mod

	@property
	def mod(self):
		return self._mod
	def __add__(self, zn):
		if isinstance(zn, Zn) and zn.mod==Zn.mod:
			return Zn(self.q+zn.q)
		else:
			raise TypeError("Cannot add quantum numbers of differe types!")

	def __neg__(self):
		self._q=(-self._q+self._mod) % self._mod
