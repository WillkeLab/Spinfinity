# TODO List

## Next
- [x] Add alternative way to IETS calculation when anyway no SpinPumping is required 
- [x] Fix the labeling and amplitude sign of the eigenvectors --> follow the typical convention 
- [ ] Test the code for various systems
- [ ] Add examples like two spin 1/2 for ESR or Fe for IETS 
- [x] Add print parameters function 
- [x] Fix: overflow encountered in exp --> somehow the fixing options makes it slower
- [ ] Structure the class better and add descriptions --> via sub classes??
- [x] Include the spin pumping in the ESR experiment
- [x] Add showing Matrix representation 

To tidy up the code, we can leave some functions for IETS but we could include them in one master function, that then decides whether to include spin pumping or not, one could also think of directly plotting the spectra for all spins
Another Idea would be to split the matrix element into parts, to later see which signal comes from elastic or inelastic parts

## Ideas
- [x] Improve performance
- [ ] Make Subclasses
- [ ] Make it more save by checking for inputs
- [ ] Make it smart that it knows what it has to calculate next, State Management, Parameter file to quickly load parameters, Logging,

