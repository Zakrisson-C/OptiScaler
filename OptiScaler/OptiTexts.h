#pragma once

// OptiKeys (ProjectID, provider IDs, FSR_* NGX parameter keys) lives in OptiTypes.h on master. The fork's
// copy of this file also carried its own OptiKeys block, which redefined those constants in any translation
// unit that saw both headers (transplant fix, 23 Sep). Only the FSR-RR display string remains here.

namespace OptiTexts
{
using CString = const char[];

// User friendly name for FSR-RR backend
inline constexpr CString FSR_RR_Name = "FSR Ray Regeneration";
} // namespace OptiTexts