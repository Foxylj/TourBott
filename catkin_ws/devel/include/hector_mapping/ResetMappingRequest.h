// Generated by gencpp from file hector_mapping/ResetMappingRequest.msg
// DO NOT EDIT!


#ifndef HECTOR_MAPPING_MESSAGE_RESETMAPPINGREQUEST_H
#define HECTOR_MAPPING_MESSAGE_RESETMAPPINGREQUEST_H


#include <string>
#include <vector>
#include <map>

#include <ros/types.h>
#include <ros/serialization.h>
#include <ros/builtin_message_traits.h>
#include <ros/message_operations.h>

#include <geometry_msgs/Pose.h>

namespace hector_mapping
{
template <class ContainerAllocator>
struct ResetMappingRequest_
{
  typedef ResetMappingRequest_<ContainerAllocator> Type;

  ResetMappingRequest_()
    : initial_pose()  {
    }
  ResetMappingRequest_(const ContainerAllocator& _alloc)
    : initial_pose(_alloc)  {
  (void)_alloc;
    }



   typedef  ::geometry_msgs::Pose_<ContainerAllocator>  _initial_pose_type;
  _initial_pose_type initial_pose;





  typedef boost::shared_ptr< ::hector_mapping::ResetMappingRequest_<ContainerAllocator> > Ptr;
  typedef boost::shared_ptr< ::hector_mapping::ResetMappingRequest_<ContainerAllocator> const> ConstPtr;

}; // struct ResetMappingRequest_

typedef ::hector_mapping::ResetMappingRequest_<std::allocator<void> > ResetMappingRequest;

typedef boost::shared_ptr< ::hector_mapping::ResetMappingRequest > ResetMappingRequestPtr;
typedef boost::shared_ptr< ::hector_mapping::ResetMappingRequest const> ResetMappingRequestConstPtr;

// constants requiring out of line definition



template<typename ContainerAllocator>
std::ostream& operator<<(std::ostream& s, const ::hector_mapping::ResetMappingRequest_<ContainerAllocator> & v)
{
ros::message_operations::Printer< ::hector_mapping::ResetMappingRequest_<ContainerAllocator> >::stream(s, "", v);
return s;
}


template<typename ContainerAllocator1, typename ContainerAllocator2>
bool operator==(const ::hector_mapping::ResetMappingRequest_<ContainerAllocator1> & lhs, const ::hector_mapping::ResetMappingRequest_<ContainerAllocator2> & rhs)
{
  return lhs.initial_pose == rhs.initial_pose;
}

template<typename ContainerAllocator1, typename ContainerAllocator2>
bool operator!=(const ::hector_mapping::ResetMappingRequest_<ContainerAllocator1> & lhs, const ::hector_mapping::ResetMappingRequest_<ContainerAllocator2> & rhs)
{
  return !(lhs == rhs);
}


} // namespace hector_mapping

namespace ros
{
namespace message_traits
{





template <class ContainerAllocator>
struct IsFixedSize< ::hector_mapping::ResetMappingRequest_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::hector_mapping::ResetMappingRequest_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::hector_mapping::ResetMappingRequest_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::hector_mapping::ResetMappingRequest_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct HasHeader< ::hector_mapping::ResetMappingRequest_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct HasHeader< ::hector_mapping::ResetMappingRequest_<ContainerAllocator> const>
  : FalseType
  { };


template<class ContainerAllocator>
struct MD5Sum< ::hector_mapping::ResetMappingRequest_<ContainerAllocator> >
{
  static const char* value()
  {
    return "3423647d14c6c84592eef8b1184a5974";
  }

  static const char* value(const ::hector_mapping::ResetMappingRequest_<ContainerAllocator>&) { return value(); }
  static const uint64_t static_value1 = 0x3423647d14c6c845ULL;
  static const uint64_t static_value2 = 0x92eef8b1184a5974ULL;
};

template<class ContainerAllocator>
struct DataType< ::hector_mapping::ResetMappingRequest_<ContainerAllocator> >
{
  static const char* value()
  {
    return "hector_mapping/ResetMappingRequest";
  }

  static const char* value(const ::hector_mapping::ResetMappingRequest_<ContainerAllocator>&) { return value(); }
};

template<class ContainerAllocator>
struct Definition< ::hector_mapping::ResetMappingRequest_<ContainerAllocator> >
{
  static const char* value()
  {
    return "geometry_msgs/Pose initial_pose\n"
"\n"
"================================================================================\n"
"MSG: geometry_msgs/Pose\n"
"# A representation of pose in free space, composed of position and orientation. \n"
"Point position\n"
"Quaternion orientation\n"
"\n"
"================================================================================\n"
"MSG: geometry_msgs/Point\n"
"# This contains the position of a point in free space\n"
"float64 x\n"
"float64 y\n"
"float64 z\n"
"\n"
"================================================================================\n"
"MSG: geometry_msgs/Quaternion\n"
"# This represents an orientation in free space in quaternion form.\n"
"\n"
"float64 x\n"
"float64 y\n"
"float64 z\n"
"float64 w\n"
;
  }

  static const char* value(const ::hector_mapping::ResetMappingRequest_<ContainerAllocator>&) { return value(); }
};

} // namespace message_traits
} // namespace ros

namespace ros
{
namespace serialization
{

  template<class ContainerAllocator> struct Serializer< ::hector_mapping::ResetMappingRequest_<ContainerAllocator> >
  {
    template<typename Stream, typename T> inline static void allInOne(Stream& stream, T m)
    {
      stream.next(m.initial_pose);
    }

    ROS_DECLARE_ALLINONE_SERIALIZER
  }; // struct ResetMappingRequest_

} // namespace serialization
} // namespace ros

namespace ros
{
namespace message_operations
{

template<class ContainerAllocator>
struct Printer< ::hector_mapping::ResetMappingRequest_<ContainerAllocator> >
{
  template<typename Stream> static void stream(Stream& s, const std::string& indent, const ::hector_mapping::ResetMappingRequest_<ContainerAllocator>& v)
  {
    s << indent << "initial_pose: ";
    s << std::endl;
    Printer< ::geometry_msgs::Pose_<ContainerAllocator> >::stream(s, indent + "  ", v.initial_pose);
  }
};

} // namespace message_operations
} // namespace ros

#endif // HECTOR_MAPPING_MESSAGE_RESETMAPPINGREQUEST_H
