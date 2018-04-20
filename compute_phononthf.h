/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing authors: Tianli Feng, Yang Zhong, and Xiulin Ruan
------------------------------------------------------------------------- */

#ifdef COMPUTE_CLASS

ComputeStyle(phononthf,ComputePHONONTHF)

#else

#ifndef LMP_COMPUTE_PHONONTHF_H
#define LMP_COMPUTE_PHONONTHF_H

#ifndef __EIGEN__
#define __EIGEN__
typedef struct eigen{
	double re[3];
	double im[3];
  } EIGEN;
#endif

#include "compute.h"

namespace LAMMPS_NS {

class ComputePHONONTHF : public Compute {
 public:
  ComputePHONONTHF(class LAMMPS *, int, char **);
  virtual ~ComputePHONONTHF();
  virtual void init();
  virtual void compute_array();

 protected:
  double pfactor;
  double efactor;
  double tfactor;

 private:
  int nk, nkpp, nkst, nked;   // Number definition for k-points
  int natoms, nbasis;
  bigint ntot;
  char *id_ke,*id_pe,*id_stress;
  class Compute *c_ke,*c_pe,*c_stress;
  EIGEN *eigenv;
  int *at2bs;
  double **kpt, **ratom;
  };

}

#endif
#endif
