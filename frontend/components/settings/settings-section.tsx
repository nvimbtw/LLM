import { GROUPS, SETTINGS } from "@/lib/settings"
import { SettingRow } from "./setting-row"
import { ThemeRow } from "./theme-row"

export interface SettingsSectionProps {
  group: (typeof GROUPS)[number]
}

export function SettingsSection({ group }: SettingsSectionProps) {
  const rows = SETTINGS.filter((setting) => setting.group === group.id)

  return (
    <section>
      <h2 className="font-display text-2xl">{group.label}</h2>
      <div className="mt-8 flex flex-col gap-6">
        {group.id === "appearance" && <ThemeRow />}
        {rows.map((setting) => (
          <SettingRow key={setting.key} setting={setting} />
        ))}
      </div>
    </section>
  )
}
