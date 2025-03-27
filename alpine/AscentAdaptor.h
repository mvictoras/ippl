#ifndef ASECNTACTORADAPTOR_H
#define ASECNTACTORADAPTOR_H

#include "Ippl.h"

#include <ascent.hpp>
#include <iostream>
#include <optional>
#include <string>
#include <vector>
#include <variant>   // Ensure std::variant is available

#include "Utility/IpplException.h"

namespace AscentAdaptor {

    // Global variables
    ascent::Ascent mAscent;
    int mFrequency = 1;

    template <typename T, unsigned Dim>
    using FieldVariant = std::variant<Field_t<Dim>*, VField_t<T, Dim>*>;
    
    template <typename T, unsigned Dim>
    using FieldPair = std::pair<std::string, FieldVariant<T, Dim>>;
    
    template <typename T, unsigned Dim>
    using ParticlePair = std::pair<std::string, std::shared_ptr<ParticleContainer<T, Dim>>>;

    using View_vector =
        Kokkos::View<ippl::Vector<double, 3>***, Kokkos::LayoutLeft, Kokkos::HostSpace>;
    inline void setData(conduit::Node& node, const View_vector& view, const std::string& fieldName) {
        node["electrostatic/association"].set_string("element");
        node["electrostatic/topology"].set_string(fieldName + "_mesh");
        node["electrostatic/volume_dependent"].set_string("false");

        auto length = std::size(view);
        node["electrostatic/values/x"].set_external(&view.data()[0][0], length, 0, 1);
        node["electrostatic/values/y"].set_external(&view.data()[0][1], length, 0, 1);
        node["electrostatic/values/z"].set_external(&view.data()[0][2], length, 0, 1);
    }

    using View_scalar = Kokkos::View<double***, Kokkos::LayoutLeft, Kokkos::HostSpace>;
    inline void setData(conduit::Node& node, const View_scalar& view, const std::string& fieldName) {
        node["density/association"].set_string("element");
        node["density/topology"].set_string(fieldName + "_mesh");
        node["density/volume_dependent"].set_string("false");

        node["density/values"].set_external(view.data(), view.size());
    }

    void Initialize(int frequency) {
      MPI_Comm ascent_comm;
      mFrequency = frequency;
      MPI_Comm_dup(MPI_COMM_WORLD, &ascent_comm);
      conduit::Node ascent_opts;
      ascent_opts["mpi_comm"] = MPI_Comm_c2f(ascent_comm);
      mAscent.open(ascent_opts);
    }

    void Finalize() {
        conduit::Node node;
        mAscent.close();
    }

    // Fixed template parameter list (removed extra comma)
    template <typename T, unsigned Dim,
              typename RHostType = typename ippl::ParticleAttrib<ippl::Vector<double, 3>>::HostMirror,
              typename PHostType = typename ippl::ParticleAttrib<ippl::Vector<double, 3>>::HostMirror,
              typename QHostType = typename ippl::ParticleAttrib<double>::HostMirror>
    void Execute_Particle(
         const std::shared_ptr<ParticleContainer<T, Dim>>& particleContainer,
         const RHostType& R_host, const PHostType& P_host, const QHostType& q_host,
         const std::string& particlesName,
         conduit::Node& node) {

        node["coordsets/" + particlesName + "_coords/type"].set_string("explicit");
        node["coordsets/" + particlesName + "_coords/values/x"].set_external(&R_host.data()[0][0],
              particleContainer->getLocalNum(), 0, sizeof(double)*3);
        node["coordsets/" + particlesName + "_coords/values/y"].set_external(&R_host.data()[0][1],
              particleContainer->getLocalNum(), 0, sizeof(double)*3);
        node["coordsets/" + particlesName + "_coords/values/z"].set_external(&R_host.data()[0][2],
              particleContainer->getLocalNum(), 0, sizeof(double)*3);

        node["topologies/" + particlesName + "_topo/type"].set_string("points");
        node["topologies/" + particlesName + "_topo/coordset"].set_string(particlesName + "_coords");

        // (Duplicate topology block - can be removed if redundant)
        node["topologies/" + particlesName + "_topo/type"].set_string("points");
        node["topologies/" + particlesName + "_topo/coordset"].set_string(particlesName + "_coords");

        // Particle center (dummy field)
        //std::vector<double> dummy_field = { 1.0 };
        auto &node_fields = node["fields"];  // renamed from "fields" to "node_fields"
        //node_fields[particlesName + "_center/association"].set_string("vertex");
        //node_fields[particlesName + "_center/topology"].set_string(particlesName + "_center_topo");
        //node_fields[particlesName + "_center/volume_dependent"].set_string("false");
        //node_fields[particlesName + "_center/values"].set(dummy_field);

        // Scalar charge field
        node_fields[particlesName + "_charge/association"].set_string("vertex");
        node_fields[particlesName + "_charge/topology"].set_string(particlesName + "_topo");
        node_fields[particlesName + "_charge/volume_dependent"].set_string("false");
        node_fields[particlesName + "_charge/values"].set_external(q_host.data(),
              particleContainer->getLocalNum());

        // Vector velocity field
        node_fields[particlesName + "_velocity/association"].set_string("vertex");
        node_fields[particlesName + "_velocity/topology"].set_string(particlesName + "_topo");
        node_fields[particlesName + "_velocity/volume_dependent"].set_string("false");
        node_fields[particlesName + "_velocity/values/x"].set_external(&P_host.data()[0][0],
              particleContainer->getLocalNum(), 0, sizeof(double)*3);
        node_fields[particlesName + "_velocity/values/y"].set_external(&P_host.data()[0][1],
              particleContainer->getLocalNum(), 0, sizeof(double)*3);
        node_fields[particlesName + "_velocity/values/z"].set_external(&P_host.data()[0][2],
              particleContainer->getLocalNum(), 0, sizeof(double)*3);

        // Position field
        node_fields[particlesName + "_position/association"].set_string("vertex");
        node_fields[particlesName + "_position/topology"].set_string(particlesName + "_topo");
        node_fields[particlesName + "_position/volume_dependent"].set_string("false");
        node_fields[particlesName + "_position/values/x"].set_external(&R_host.data()[0][0],
              particleContainer->getLocalNum(), 0, sizeof(double)*3);
        node_fields[particlesName + "_position/values/y"].set_external(&R_host.data()[0][1],
              particleContainer->getLocalNum(), 0, sizeof(double)*3);
        node_fields[particlesName + "_position/values/z"].set_external(&R_host.data()[0][2],
              particleContainer->getLocalNum(), 0, sizeof(double)*3);

        // (Magnitude field code is commented out)

        conduit::Node verify_info;
        if(!conduit::blueprint::mesh::verify(node, verify_info))
        {
            std::cerr << "Mesh verification failed!" << std::endl;
            verify_info.print();
            exit(EXIT_FAILURE);
        }
    }

    template <class Field>
    void Execute_Field(Field* field, const std::string& fieldName,
         Kokkos::View<typename Field::view_type::data_type, Kokkos::LayoutLeft, Kokkos::HostSpace>& host_view_layout_left,
         conduit::Node& node) {
        static_assert(Field::dim == 3, "AscentAdaptor only supports 3D");

        node["coordsets/" + fieldName + "_coords/type"].set_string("uniform");

        std::string field_node_dim{"coordsets/" + fieldName + "_coords/dims/i"};
        std::string field_node_origin{"coordsets/" + fieldName + "_coords/origin/x"};
        std::string field_node_spacing{"coordsets/" + fieldName + "_coords/spacing/dx"};

        for (unsigned int iDim = 0; iDim < field->get_mesh().getGridsize().dim; ++iDim) {
            node[field_node_dim].set(field->getLayout().getLocalNDIndex()[iDim].length() + 1);
            node[field_node_origin].set(
                field->get_mesh().getOrigin()[iDim] +
                field->getLayout().getLocalNDIndex()[iDim].first() *
                field->get_mesh().getMeshSpacing(iDim));
            node[field_node_spacing].set(field->get_mesh().getMeshSpacing(iDim));

            ++field_node_dim.back();
            ++field_node_origin.back();
            ++field_node_spacing.back();
        }

        node["topologies/" + fieldName + "_mesh/type"].set_string("uniform");
        node["topologies/" + fieldName + "_mesh/coordset"].set_string(fieldName + "_coords");
        std::string field_node_origin_topo{"topologies/" + fieldName + "_mesh/origin/x"};
        for (unsigned int iDim = 0; iDim < field->get_mesh().getGridsize().dim; ++iDim) {
            node[field_node_origin_topo].set(
                field->get_mesh().getOrigin()[iDim] +
                field->getLayout().getLocalNDIndex()[iDim].first() *
                field->get_mesh().getMeshSpacing(iDim));
            ++field_node_origin_topo.back();
        }

        host_view_layout_left = Kokkos::View<typename Field::view_type::data_type,
            Kokkos::LayoutLeft, Kokkos::HostSpace>(
           "host_view_layout_left",
           field->getLayout().getLocalNDIndex()[0].length(),
           field->getLayout().getLocalNDIndex()[1].length(),
           field->getLayout().getLocalNDIndex()[2].length());

        auto host_view =
            Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), field->getView());

        auto nGhost = field->getNghost();
        for (size_t i = 0; i < field->getLayout().getLocalNDIndex()[0].length(); ++i) {
            for (size_t j = 0; j < field->getLayout().getLocalNDIndex()[1].length(); ++j) {
                for (size_t k = 0; k < field->getLayout().getLocalNDIndex()[2].length(); ++k) {
                    host_view_layout_left(i, j, k) = host_view(i + nGhost, j + nGhost, k + nGhost);
                }
            }
        }

        auto &node_fields = node["fields"];  // renamed to avoid conflict
        setData(node_fields, host_view_layout_left, fieldName);

        conduit::Node verify_info;
        if(!conduit::blueprint::mesh::verify(node, verify_info))
        {
            std::cerr << "Mesh verification failed!" << std::endl;
            verify_info.print();
            exit(EXIT_FAILURE);
        }
    }

    template<typename T, unsigned Dim>
    void Execute(int cycle, double time,
         const std::vector<ParticlePair<T, Dim>>& particles,
         const std::vector<FieldPair<T, Dim>>& fields) {
        conduit::Node node;

        if((cycle+1) % mFrequency != 0) return;

        auto state = node["state"];
        state["cycle"].set(cycle);
        state["time"].set(time);

        std::map<std::string, typename ippl::ParticleAttrib<ippl::Vector<double, 3>>::HostMirror> R_host_map;
        std::map<std::string, typename ippl::ParticleAttrib<ippl::Vector<double, 3>>::HostMirror> P_host_map;
        std::map<std::string, typename ippl::ParticleAttrib<ippl::Vector<double, 3>>::HostMirror> center_host_map;
        std::map<std::string, typename ippl::ParticleAttrib<double>::HostMirror> q_host_map;
        std::map<std::string, typename ippl::ParticleAttrib<std::int64_t>::HostMirror> ID_host_map;

        for (const auto& particlesPair : particles)
        {
            const std::string& particlesName = particlesPair.first;
            const auto particleContainer = particlesPair.second;

            assert((particleContainer->ID.getView().data() != nullptr) &&
                   "ID view should not be nullptr, might be missing the right execution space");

            R_host_map[particlesName] = particleContainer->R.getHostMirror();
            P_host_map[particlesName] = particleContainer->P.getHostMirror();
            q_host_map[particlesName] = particleContainer->q.getHostMirror();

            Kokkos::deep_copy(R_host_map[particlesName], particleContainer->R.getView());
            Kokkos::deep_copy(P_host_map[particlesName], particleContainer->P.getView());
            Kokkos::deep_copy(q_host_map[particlesName], particleContainer->q.getView());

            Execute_Particle(
              particleContainer,
              R_host_map[particlesName], P_host_map[particlesName], q_host_map[particlesName],
              particlesName,
              node);
        }

        std::map<std::string, Kokkos::View<typename Field_t<Dim>::view_type::data_type,
            Kokkos::LayoutLeft, Kokkos::HostSpace>> scalar_host_views;
        std::map<std::string, Kokkos::View<typename VField_t<T, Dim>::view_type::data_type,
            Kokkos::LayoutLeft, Kokkos::HostSpace>> vector_host_views;

        for (const auto& fieldPair : fields)
        {
            const std::string& fieldName = fieldPair.first;
            const auto& fieldVariant = fieldPair.second;

            if (std::holds_alternative<Field_t<Dim>*>(fieldVariant)) {
                Field_t<Dim>* field = std::get<Field_t<Dim>*>(fieldVariant);
                Execute_Field(field, fieldName, scalar_host_views[fieldName], node);
            }
            else if (std::holds_alternative<VField_t<T, Dim>*>(fieldVariant)) {
                VField_t<T, Dim>* field = std::get<VField_t<T, Dim>*>(fieldVariant);
                Execute_Field(field, fieldName, vector_host_views[fieldName], node);
            }
        }

        conduit::Node actions;
        mAscent.publish(node);
        mAscent.execute(actions);
    }
}  // namespace AscentAdaptor

#endif

