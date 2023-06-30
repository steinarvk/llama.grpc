#ifndef H_LLAMAGRPC_STORAGE
#define H_LLAMAGRPC_STORAGE

#include <string>
#include <filesystem>

#include "absl/status/status.h"
#include "absl/status/statusor.h"

#include "proto/internal/llamagrpc_internal.pb.h"
#include "sqlite3.h"

namespace llamagrpc {

class Storage {
private:
    std::string database_filename;
    std::filesystem::path checkpoint_path;
    sqlite3 *db;

    bool Wrap(int rc);
    void ExecOrDie(const char* sql);

    Storage();

    void OpenDatabaseOrDie();
    void InitSchemaOrDie();
    int GetSchemaVersionOrDie();
    bool SchemaUpgradeIfNecessaryOrDie();

    static int CURRENT_SCHEMA_VERSION;

public:
    static Storage* CreateFromFlags();

    absl::Status RegisterSnapshot(const llamagrpc_internal::SnapshotDesc& snapshot_desc);

    ~Storage();
};

}

#endif