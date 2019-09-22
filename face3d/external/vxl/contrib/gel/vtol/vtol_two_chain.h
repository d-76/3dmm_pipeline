// This is gel/vtol/vtol_two_chain.h
#ifndef vtol_two_chain_h_
#define vtol_two_chain_h_
//-----------------------------------------------------------------------------
//:
// \file
// \brief Represents a connected chain of faces
//
// The vtol_two_chain class is used to represent a set of faces on a topological
//  structure. A vtol_two_chain consists of its inferior faces and the superiors
//  on which it lies.  A vtol_two_chain may or may not be an ordered cycle.  If
//  the chain of faces encloses a volume, then the vtol_two_chain may be used as
//  the boundary of a topological vtol_block in a 3D structure.
//
// \author
//     Patricia A. Vrobel
//
// \verbatim
//  Modifications:
//   JLM December 1995, Added timeStamp (Touch) to
//       operations which affect bounds.
//   JLM December 1995, no local method for ComputeBoundingBox
//       Should use vtol_face geometry recursively to be proper.
//       Currently reverts to bounds on vertices from
//       vtol_topology_object::ComputeBoundingBox()
//   05/13/98  RIH replaced append by insert_after to avoid n^2 behavior
//   PTU May 2000 ported to vxl
//   Dec. 2002, Peter Vanroose -interface change: vtol objects -> smart pointers
// \endverbatim
//-----------------------------------------------------------------------------

#include <iostream>
#include <iosfwd>
#include <vector>
#ifdef _MSC_VER
#  include <vcl_msvc_warnings.h>
#endif
#include <vtol/vtol_chain.h>
#include <vtol/vtol_face_2d_sptr.h>
class vtol_vertex;
class vtol_edge;
class vtol_zero_chain;
class vtol_one_chain;
class vtol_face;
class vtol_block;

class vtol_two_chain : public vtol_chain
{
 public:
  //***************************************************************************
  // Initialization
  //***************************************************************************

  void link_chain_inferior(const vtol_two_chain_sptr& chain_inferior);
  void unlink_chain_inferior(const vtol_two_chain_sptr& chain_inferior);

  //---------------------------------------------------------------------------
  //: Default constructor
  //---------------------------------------------------------------------------
  vtol_two_chain() { is_cycle_=false; }

  //---------------------------------------------------------------------------
  //: Constructor
  //---------------------------------------------------------------------------
  explicit vtol_two_chain(int /*num_faces*/) { is_cycle_=false; }

  //---------------------------------------------------------------------------
  //: Constructor
  //---------------------------------------------------------------------------
  explicit vtol_two_chain(face_list const&, bool new_is_cycle=false);

  //---------------------------------------------------------------------------
  //: Constructor
  //---------------------------------------------------------------------------
  vtol_two_chain(face_list const&,
                 std::vector<signed char> const&,
                 bool new_is_cycle=false);

  //---------------------------------------------------------------------------
  //: Pseudo copy constructor.  Deep copy.
  //---------------------------------------------------------------------------
  vtol_two_chain(vtol_two_chain_sptr const& other);
 private:
  //---------------------------------------------------------------------------
  //: Copy constructor.  Deep copy.  Deprecated.
  //---------------------------------------------------------------------------
  vtol_two_chain(vtol_two_chain const& other) = delete;
 public:
  //---------------------------------------------------------------------------
  //: Destructor
  //---------------------------------------------------------------------------
  ~vtol_two_chain() override;

  //---------------------------------------------------------------------------
  //: Clone `this': creation of a new object and initialization
  //  See Prototype pattern
  //---------------------------------------------------------------------------
  vsol_spatial_object_2d* clone() const override;

  //: Return a platform independent string identifying the class
  std::string is_a() const override { return std::string("vtol_two_chain"); }

  //: Return true if the argument matches the string identifying the class or any parent class
  bool is_class(const std::string& cls) const override
  { return cls==is_a() || vtol_chain::is_class(cls); }

  virtual vtol_two_chain * copy_with_arrays(topology_list &verts,
                                            topology_list &edges) const;
  // Accessors

 private: // has been superseded by is_a()
  //: Return the topology type
  vtol_topology_object_type topology_type() const override {return TWOCHAIN;}

 public:
  //: get the direction of the face
  signed char direction(vtol_face const& f) const;

  virtual vtol_face_sptr face(int i);

  //---------------------------------------------------------------------------
  //: Shallow copy with no links
  //---------------------------------------------------------------------------
  virtual vtol_topology_object *shallow_copy_with_no_links() const;

  virtual void add_superiors_from_parent(topology_list &);
  virtual void remove_superiors_of_parent(topology_list &);
  virtual void remove_superiors();
  virtual void update_superior_list_p_from_hierarchy_parent();

  virtual void add_face(vtol_face_sptr const&, signed char);
  virtual void remove_face(vtol_face_sptr const&);
 private:
  // Deprecated:
  virtual void add_face(vtol_face &,signed char);
  virtual void remove_face(vtol_face &);
 public:
  //***************************************************************************
  // Replaces dynamic_cast<T>
  //***************************************************************************

  //---------------------------------------------------------------------------
  //: Return `this' if `this' is a two_chain, 0 otherwise
  //---------------------------------------------------------------------------
  const vtol_two_chain *cast_to_two_chain() const override { return this; }

  //---------------------------------------------------------------------------
  //: Return `this' if `this' is a two_chain, 0 otherwise
  //---------------------------------------------------------------------------
  vtol_two_chain *cast_to_two_chain() override { return this; }

  //***************************************************************************
  // Status report
  //***************************************************************************

  void link_inferior(const vtol_face_sptr& inf);
  void unlink_inferior(const vtol_face_sptr& inf);

  //---------------------------------------------------------------------------
  //: Is `inferior' type valid for `this' ?
  //---------------------------------------------------------------------------
  bool valid_inferior_type(vtol_topology_object const* inferior) const override
  { return inferior->cast_to_face()!=nullptr; }
  bool valid_inferior_type(vtol_face_sptr const& )    const { return true; }
  bool valid_inferior_type(vtol_face_2d_sptr const& ) const { return true; }
  bool valid_superior_type(vtol_block_sptr const& )   const { return true; }

  //---------------------------------------------------------------------------
  //: Is `chain_inf_sup' type valid for `this' ?
  //---------------------------------------------------------------------------
  bool valid_chain_type(vtol_chain_sptr chain_inf_sup) const override
  { return chain_inf_sup->cast_to_two_chain()!=nullptr; }
  bool valid_chain_type(vtol_two_chain_sptr const& ) const { return true; }

  // network access methods

  virtual vertex_list *outside_boundary_vertices();
  virtual zero_chain_list *outside_boundary_zero_chains();
  virtual edge_list *outside_boundary_edges();
  virtual one_chain_list *outside_boundary_one_chains();
  virtual face_list *outside_boundary_faces();
  virtual two_chain_list *outside_boundary_two_chains();

  // The returned pointers must be deleted after use.
  virtual two_chain_list *inferior_two_chains();
  // The returned pointers must be deleted after use.
  virtual two_chain_list *superior_two_chains();

 protected:
  // \warning these methods should not be used by clients
  // The returned pointers must be deleted after use.

  std::vector<vtol_vertex*> *compute_vertices() override;
  std::vector<vtol_edge*> *compute_edges() override;
  std::vector<vtol_zero_chain*> *compute_zero_chains() override;
  std::vector<vtol_one_chain*> *compute_one_chains() override;
  std::vector<vtol_face*> *compute_faces() override;
  std::vector<vtol_two_chain*> *compute_two_chains() override;
  std::vector<vtol_block*> *compute_blocks() override;

 public:
  virtual std::vector<vtol_vertex*> *outside_boundary_compute_vertices();
  virtual std::vector<vtol_zero_chain*> *outside_boundary_compute_zero_chains();
  virtual std::vector<vtol_edge*> *outside_boundary_compute_edges();
  virtual std::vector<vtol_one_chain*> *outside_boundary_compute_one_chains();
  virtual std::vector<vtol_face*> *outside_boundary_compute_faces();
  virtual std::vector<vtol_two_chain*> *outside_boundary_compute_two_chains();

  int num_faces() const { return numinf(); }

  virtual void correct_chain_directions();

  virtual bool operator==(vtol_two_chain const& other) const;
  inline bool operator!=(const vtol_two_chain &other)const{return !operator==(other);}
  bool operator==(vsol_spatial_object_2d const& obj) const override; // virtual of vsol_spatial_object_2d

  void print(std::ostream &strm=std::cout) const override;
  virtual void describe_directions(std::ostream &strm=std::cout, int blanking=0) const;
  void describe(std::ostream &strm=std::cout, int blanking=0) const override;

  virtual bool break_into_connected_components(topology_list &components);
};

#endif // vtol_two_chain_h_
