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

#include <mpi.h>
#include "stdlib.h"
#include <string.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include "atom.h"
#include "update.h"
#include "modify.h"
#include "force.h"
#include "error.h"
#include "group.h"
#include "memory.h"
#include "compute_phononthf.h"

using namespace std;
using namespace LAMMPS_NS;

#define INVOKED_PERATOM 8

//***********************************************************************************************************************

ComputePHONONTHF::ComputePHONONTHF(LAMMPS *lmp, int narg, char **arg) :
	Compute(lmp, narg, arg),
	id_ke(NULL), id_pe(NULL), id_stress(NULL),
	kpt(NULL), eigenv(NULL), at2bs(NULL), ratom(NULL)
{
	if (narg != 7) error->all(FLERR,"Illegal compute phononthf command");
	array_flag = 1;
	extarray = 0;
  	dynamic_group_allow = 0;

	nbasis = force->inumeric(FLERR,arg[3]);
  	if (nbasis < 1) error->all(FLERR,"Illegal compute phononthf command");

  // store ke/atom, pe/atom, stress/atom IDs used by phononthf computation
  // insure they are valid for these computations

	int n = strlen(arg[4]) + 1;
	id_ke = new char[n];
	strcpy(id_ke,arg[4]);

	n = strlen(arg[5]) + 1;
	id_pe = new char[n];
	strcpy(id_pe,arg[5]);

	n = strlen(arg[6]) + 1;
	id_stress = new char[n];
	strcpy(id_stress,arg[6]);

	int ike = modify->find_compute(id_ke);
	int ipe = modify->find_compute(id_pe);
	int istress = modify->find_compute(id_stress);
	if (ike < 0 || ipe < 0 || istress < 0)
		error->all(FLERR,"Could not find compute phononthf compute ID");
	if (strcmp(modify->compute[ike]->style,"ke/atom") != 0)
    	error->all(FLERR,"Compute phononthf compute ID does not compute ke/atom");
	if (modify->compute[ipe]->peatomflag == 0)
    	error->all(FLERR,"Compute phononthf compute ID does not compute pe/atom");
	if (modify->compute[istress]->pressatomflag == 0)
    	error->all(FLERR,"Compute phononthf compute ID does not compute stress/atom");

	natoms = group->count(igroup);
	if (natoms <= 0)
    	error->all(FLERR,"Not valid atoms in this group");
	ntot = atom->natoms;

	int i, j, l, s, jj, tt, flag, format, prepared;
	double xtmp[3], frac[3], rec[3][3];
	char tempstr[255];

	ifstream kpnts("kpoints.txt");				//reative path ("../[source]/kpoints.txt")
	ifstream recpvec("recipvec.txt");			//reative path ("../[srouce]/recipvec.txt")
	ifstream eigen_txt("eigenvectors.txt");		//reative path ("../[srouce]/eigenvectors.txt")
	ifstream cellmap("cellbasismap.txt");		//reative path ("../[srouce]/cellbasismap.txt")
	ifstream coord("lammps.txt");				//reative path ("../[srouce]/lammps.txt")

	if (!kpnts.is_open())
		error->all(FLERR,"Could not open file: kpoints.txt");
	if (!recpvec.is_open())
		error->all(FLERR,"Could not open file: recipvec.txt");
	if (!eigen_txt.is_open())
		error->all(FLERR,"Could not open file: eigenvectors.txt");
	if (!cellmap.is_open())
		error->all(FLERR,"Could not open file: cellbasismap.txt");
	if (!coord.is_open())
		error->all(FLERR,"Could not open file: lammps.txt");

	kpnts>>nk;
	nkst = 0;
	nked = nk;
	nkpp = nked - nkst; // #kpoints per processor

	eigenv = new EIGEN [nk*(nbasis*3)*nbasis];
	at2bs = new int [ntot];	//map atom id to basis atoms
	memory->create(kpt,nk,3,"phononthf:kpt");

	for(i=0;i<3;i++)
		recpvec>>rec[0][i]>>rec[1][i]>>rec[2][i];

	for(i=0;i<nk*3*nbasis*nbasis;i++)
		  for(j=0;j<3;j++) eigen_txt>>eigenv[i].re[j]>>eigenv[i].im[j];

	for(i=0;i<nk;i++){
		for(j=0;j<3;j++){
			kpnts>>frac[j];
		}
		for(l=0;l<3;l++){
			kpt[i][l]=0;
			for(s=0;s<3;s++){
				kpt[i][l]+=rec[l][s]*frac[s];//put atoms xyz coord back to Ang for convenience
			}
		}
	}

	// Atom-basis mapping
	int id, cellid, basisid;
	for(i=0;i<ntot;i++){
		cellmap>>id>>cellid>>basisid;
		at2bs[id-1]=basisid;
	}
	//Mapping Done!

// read lammps.txt
	char *pch;
	double chg;
	int mid, ntot_input;
	coord>>ntot_input;
	if (ntot!=ntot_input)
		error->all(FLERR,"Total number of atoms from lammps.txt and LAMMPS itself are not consistent!");
	memory->create(ratom,ntot,3,"phononthf:ratom");
	i=0;// check if lammps.txt is in the right format
	while(1){
		coord.getline(tempstr,256);
		if(!(strstr(tempstr,"Atoms"))) continue;
		coord.getline(tempstr,256);
		coord.getline(tempstr,256);
		break;
	}
	pch = strtok (tempstr," ,\t");
	while(pch!=NULL){
		pch = strtok (NULL," ,\t");
		i++;
	}
	format = i;
	if( format !=5 && format !=7 ){
		error->all(FLERR,"Illegal lammps.txt format for phononthf");
		exit(0);
	}
	else{
		coord.clear();
		coord.seekg(0, ios::beg);
	}
	prepared=0;
	while(!prepared){
		coord.getline(tempstr,256);
		if(!(strstr(tempstr,"Atoms"))) continue;
		coord.getline(tempstr,256);
		for(i=0;i<ntot;i++){
			if(format == 5){
				coord>>id>>jj>>xtmp[0]>>xtmp[1]>>xtmp[2];
			}
			else{
				coord>>id>>mid>>tt>>chg>>xtmp[0]>>xtmp[1]>>xtmp[2];
			}
			for(l=0;l<3;l++){
				ratom[id-1][l]=xtmp[l]; // id starts from 0
			}
		}
		prepared=1;
	}
	//Reading lammps.txt Done!

	kpnts.close();
	recpvec.close();
	eigen_txt.close();
	cellmap.close();
	coord.close();

	size_array_rows = nk;
 	size_array_cols = ((nbasis*3) + (nbasis*3)*3);
	memory->create(array,size_array_rows,size_array_cols,"phononthf:array");

}


  ComputePHONONTHF::~ComputePHONONTHF()
{
	delete [] id_ke;
	delete [] id_pe;
	delete [] id_stress;
	delete [] eigenv;
	delete [] at2bs;
	memory->destroy(kpt);
	memory->destroy(ratom);
	memory->destroy(array);

}


  void ComputePHONONTHF::init()
{
	int ike = modify->find_compute(id_ke);
	int ipe = modify->find_compute(id_pe);
	int istress = modify->find_compute(id_stress);
	if (ike < 0 || ipe < 0 || istress < 0)
		error->all(FLERR,"Could not find compute phononthf compute ID");
	c_ke = modify->compute[ike];
	c_pe = modify->compute[ipe];
	c_stress = modify->compute[istress];

	pfactor = force->nktv2p;
	efactor = force->mvv2e;
	tfactor = 2/force->boltz;

}


void ComputePHONONTHF::compute_array()
{

	// invoke 3 computes if they haven't been already
	if (!(c_ke->invoked_flag & INVOKED_PERATOM)) {
		c_ke->compute_peratom();
		c_ke->invoked_flag |= INVOKED_PERATOM;
	}
	if (!(c_pe->invoked_flag & INVOKED_PERATOM)) {
		c_pe->compute_peratom();
		c_pe->invoked_flag |= INVOKED_PERATOM;
	}
	if (!(c_stress->invoked_flag & INVOKED_PERATOM)) {
		c_stress->compute_peratom();
		c_stress->invoked_flag |= INVOKED_PERATOM;
	}

	double *ke = c_ke->vector_atom;
  	double *pe = c_pe->vector_atom;
 	double **stress = c_stress->array_atom;
	double **v = atom->v;
	int *type = atom->type;
	int *mask = atom->mask;
  	double *mass = atom->mass;
	tagint *tag = atom->tag;
	int nlocal = atom->nlocal;

	invoked_array = update->ntimestep;

	int d,i,j,k,l,s,n,m,jj,kk;
	int ik, iv, is;
	int flag;
	int id, nv, indeigen;
	double eiqr, cs, sn, StressEigR[3], StressEigI[3], sqrtmass, sqrtmass2, xtmp[3];

	//Allocation
	double **vnf;
	vnf = new double* [ntot];
	for(i=0;i<ntot;i++) vnf[i] = new double [3];

	double **stressf;
	stressf = new double* [ntot];
	for(i=0;i<ntot;i++) stressf[i] = new double [6];

	double *engf;
	engf = new double [ntot];

	double sumvre[nkpp][nbasis*3], sumvim[nkpp][nbasis*3], hfre[nkpp][nbasis*3][3], hfim[nkpp][nbasis*3][3], Temp[nkpp][nbasis*3], energyk[nkpp][nbasis*3];

	for(ik=0;ik<nkpp;ik++){
		for(iv=0;iv<nbasis*3;iv++){
			sumvre[ik][iv]=0;
			sumvim[ik][iv]=0;
			energyk[ik][iv]=0;
			for(d=0;d<3;d++){
				hfre[ik][iv][d]=0;
				hfim[ik][iv][d]=0;
			}
		}
	}
	//Allocation Done!

	for(i=0;i<ntot;i++){
		for(j=0;j<3;j++) vnf[i][j] = 0.0;
		engf[i] = 0.0;
		for(j=0;j<6;j++) stressf[i][j] = 0.0;
	}
	for(i=0;i<ntot;i++){
		if (!(mask[i] & groupbit)) continue;
		id = tag[i];
		for(j=0;j<3;j++) vnf[id-1][j] = v[i][j];
		engf[id-1] = pe[i] + ke[i];
		for(j=0;j<6;j++) stressf[id-1][j] = stress[i][j]/pfactor;
	}

	for(ik=nkst;ik<nked;ik++){
		// loop to calculate Xdot
		for(i=0;i<ntot;i++){
			if (!(mask[i] & groupbit)) continue;
			id = tag[i];
			kk = at2bs[id-1];	// only knowledge about tag of basis atom is needed, no cell info needed.
			sqrtmass = sqrt(mass[type[id-1]]/(natoms/nbasis));	// ncell=natoms/nbasis
			eiqr=0.;
			for(d=0;d<3;d++)
				eiqr += -kpt[ik][d]*ratom[id-1][d];
			cs=cos(eiqr);
			sn=sin(eiqr);
			for(iv=0;iv<nbasis*3;iv++){
				indeigen=ik*nbasis*3*nbasis+iv*nbasis+kk;
				for(d=0;d<3;d++){
					sumvre[ik][iv] += vnf[id-1][d]*(eigenv[indeigen].re[d]*cs+eigenv[indeigen].im[d]*sn)*sqrtmass;
					sumvim[ik][iv] += vnf[id-1][d]*(eigenv[indeigen].re[d]*sn-eigenv[indeigen].im[d]*cs)*sqrtmass;
				}
			}
		}// Xdot

		// loop to calculate Heat Flux
		for(i=0;i<ntot;i++){
			if (!(mask[i] & groupbit)) continue;
			id = tag[i];
			kk = at2bs[id-1];	// only knowledge about tag of basis atom is needed, no cell info needed.
			sqrtmass2 = sqrt(1/mass[type[id-1]]/(natoms/nbasis));	//ncell=natoms/nbasis
			eiqr=0.;
			for(d=0;d<3;d++)
				eiqr += kpt[ik][d]*ratom[id-1][d];
			cs=cos(eiqr);
			sn=sin(eiqr);
			for(iv=0;iv<nbasis*3;iv++){
				indeigen = ik*nbasis*3*nbasis+iv*nbasis+kk;
				StressEigR[0] = stressf[id-1][0]*eigenv[indeigen].re[0]+stressf[id-1][3] * eigenv[indeigen].re[1]+stressf[id-1][4] * eigenv[indeigen].re[2];// Sxx*Eigenx+Sxy*Eigeny+Sxz*Eigenz
				StressEigR[1] = stressf[id-1][3]*eigenv[indeigen].re[0]+stressf[id-1][1] * eigenv[indeigen].re[1]+stressf[id-1][5] * eigenv[indeigen].re[2];// Syx*Eigenx+Syy*Eigeny+Syz*Eigenz
				StressEigR[2] = stressf[id-1][4]*eigenv[indeigen].re[0]+stressf[id-1][5] * eigenv[indeigen].re[1]+stressf[id-1][2] * eigenv[indeigen].re[2];// Szx*Eigenx+Szy*Eigeny+Szz*Eigenz
				StressEigI[0] = stressf[id-1][0]*eigenv[indeigen].im[0]+stressf[id-1][3] * eigenv[indeigen].im[1]+stressf[id-1][4] * eigenv[indeigen].im[2];// Sxx*Eigenx+Sxy*Eigeny+Sxz*Eigenz
				StressEigI[1] = stressf[id-1][3]*eigenv[indeigen].im[0]+stressf[id-1][1] * eigenv[indeigen].im[1]+stressf[id-1][5] * eigenv[indeigen].im[2];// Syx*Eigenx+Syy*Eigeny+Syz*Eigenz
				StressEigI[2] = stressf[id-1][4]*eigenv[indeigen].im[0]+stressf[id-1][5] * eigenv[indeigen].im[1]+stressf[id-1][2] * eigenv[indeigen].im[2];// Szx*Eigenx+Szy*Eigeny+Szz*Eigenz
				for(d=0;d<3;d++){
					hfre[ik][iv][d] += sqrtmass2*engf[id-1]*(eigenv[indeigen].re[d]*(cs*sumvre[ik][iv]-sn*sumvim[ik][iv]) - eigenv[indeigen].im[d]*(sn*sumvre[ik][iv]+cs*sumvim[ik][iv]));
					hfre[ik][iv][d] += -sqrtmass2*(StressEigR[d]*(cs*sumvre[ik][iv]-sn*sumvim[ik][iv]) - StressEigI[d]*(sn*sumvre[ik][iv]+cs*sumvim[ik][iv]));
					hfim[ik][iv][d] += sqrtmass2*engf[id-1]*(eigenv[indeigen].re[d]*(cs*sumvim[ik][iv]+sn*sumvre[ik][iv]) + eigenv[indeigen].im[d]*(cs*sumvre[ik][iv]-sn*sumvim[ik][iv]));
					hfim[ik][iv][d] += -sqrtmass2*(StressEigI[d]*(cs*sumvre[ik][iv]-sn*sumvim[ik][iv]) + StressEigR[d]*(sn*sumvre[ik][iv]+cs*sumvim[ik][iv]));
				}
			}
		}	// Heat Flux
		for(iv=0;iv<nbasis*3;iv++){
			energyk[ik][iv] += (sumvre[ik][iv]*sumvre[ik][iv]+sumvim[ik][iv]*sumvim[ik][iv])/2.0;
		}
		for(iv=0;iv<nbasis*3;iv++){
			sumvre[ik][iv]=0.;
			sumvim[ik][iv]=0.;
		}
	}	// for k-points

	for(ik=0;ik<nkpp;ik++){
		for(iv=0;iv<nbasis*3;iv++){
			energyk[ik][iv] = energyk[ik][iv]*efactor;
			Temp[ik][iv] = energyk[ik][iv]*tfactor;
			array[ik][iv*(3+1)] = Temp[ik][iv];
			for (d=0;d<3;d++){
				array[ik][iv*(3+1)+d+1] = hfre[ik][iv][d];
			}
		}
	}

	for (i = 0; i < ntot; i ++){
     delete [] vnf[i];
		 delete [] stressf[i];
	}
	delete [] engf;
	delete [] vnf;
	delete [] stressf;
}
