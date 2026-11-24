import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

ApplicationWindow {
    id: window

    width: 1450
    height: 930
    minimumWidth: 1120
    minimumHeight: 720
    visible: true
    title: "VIctory — Grand Theft Auto VI"
    color: "#0c0d12"
    font.family: "Segoe UI"

    property int pageIndex: 0
    readonly property color gold: "#f0cd73"
    readonly property color vicePink: "#ee9ab4"
    readonly property color ocean: "#75c6ce"
    readonly property color paper: "#f1eee9"
    readonly property color muted: "#898b8d"
    readonly property color panel: "#171820"
    readonly property color line: "#2b2c35"

    RowLayout {
        anchors.fill: parent
        spacing: 0

        Rectangle {
            id: sidebar
            Layout.preferredWidth: 226
            Layout.minimumWidth: 206
            Layout.fillHeight: true
            color: "#111218"
            border.color: "#252630"
            border.width: 1

            ColumnLayout {
                anchors.fill: parent
                anchors.leftMargin: 18
                anchors.rightMargin: 18
                anchors.topMargin: 24
                anchors.bottomMargin: 17
                spacing: 0

                RowLayout {
                    Layout.fillWidth: true
                    spacing: 11

                    Rectangle {
                        width: 38
                        height: 38
                        radius: 10
                        color: window.gold
                        Text {
                            anchors.centerIn: parent
                            text: "R*"
                            color: "#191817"
                            font.pixelSize: 16
                            font.weight: Font.Black
                            font.family: "Segoe UI Black"
                        }
                    }
                    Column {
                        spacing: 1
                        Text {
                            text: "ROCKSTAR GAMES"
                            color: "#f4f1eb"
                            font.pixelSize: 13
                            font.weight: Font.Bold
                            font.letterSpacing: 1.3
                        }
                        Text {
                            text: "GRAND THEFT AUTO VI"
                            color: "#a59baa"
                            font.pixelSize: 10
                            font.weight: Font.Bold
                            font.letterSpacing: 1.5
                        }
                    }
                    Item { Layout.fillWidth: true }
                    Rectangle {
                        width: 7
                        height: 7
                        radius: 9
                    color: window.vicePink
                    }
                }

                Rectangle {
                    Layout.fillWidth: true
                    Layout.topMargin: 26
                    Layout.bottomMargin: 23
                    height: 1
                    color: "#292b2d"
                }

                Text {
                    Layout.leftMargin: 9
                    text: "ROCKSTAR  /  GTA VI"
                    color: "#686b6e"
                    font.pixelSize: 11
                    font.weight: Font.Bold
                    font.letterSpacing: 1.8
                }

                ColumnLayout {
                    Layout.fillWidth: true
                    Layout.topMargin: 11
                    spacing: 4

                    Repeater {
                        model: [
                            { label: "GTA VI", icon: "▦" },
                            { label: "Activity", icon: "◷" },
                            { label: "Settings", icon: "⚙" }
                        ]

                        delegate: Rectangle {
                            required property var modelData
                            required property int index
                            Layout.fillWidth: true
                            height: 43
                            radius: 9
                            color: window.pageIndex === index ? "#302831" : navMouse.containsMouse ? "#202129" : "transparent"

                            Rectangle {
                                visible: window.pageIndex === index
                                width: 2
                                height: 19
                                radius: 1
                                anchors.left: parent.left
                                anchors.verticalCenter: parent.verticalCenter
                                color: window.gold
                            }
                            RowLayout {
                                anchors.fill: parent
                                anchors.leftMargin: 14
                                anchors.rightMargin: 12
                                spacing: 12
                                Text {
                                    text: modelData.icon
                                    color: window.pageIndex === index ? window.gold : "#8d8f90"
                                    font.pixelSize: 16
                                    Layout.preferredWidth: 19
                                    horizontalAlignment: Text.AlignHCenter
                                }
                                Text {
                                    text: modelData.label
                                    color: window.pageIndex === index ? "#f2eee8" : "#a7a8a8"
                                    font.pixelSize: 14
                                    font.weight: window.pageIndex === index ? Font.DemiBold : Font.Normal
                                    Layout.fillWidth: true
                                }
                                Text {
                                    visible: index === 0 && launcher.hasGame
                                    text: "1"
                                    color: "#b6a38e"
                                    font.pixelSize: 12
                                }
                            }
                            MouseArea {
                                id: navMouse
                                anchors.fill: parent
                                hoverEnabled: true
                                onClicked: window.pageIndex = index
                            }
                        }
                    }
                }

                Rectangle {
                    Layout.fillWidth: true
                    Layout.topMargin: 20
                    Layout.bottomMargin: 15
                    height: 1
                    color: "#292b2d"
                }

                Text {
                    Layout.leftMargin: 9
                    text: "GTA VI  /  GAME FILES"
                    color: "#686b6e"
                    font.pixelSize: 11
                    font.weight: Font.Bold
                    font.letterSpacing: 1.6
                }

                Rectangle {
                    Layout.fillWidth: true
                    Layout.topMargin: 10
                    height: 57
                    radius: 10
                    color: "#191a23"
                    border.color: "#2b2c36"
                    RowLayout {
                        anchors.fill: parent
                        anchors.leftMargin: 8
                        anchors.rightMargin: 9
                        spacing: 9
                        Rectangle {
                            width: 38
                            height: 41
                            radius: 7
                            gradient: Gradient {
                                GradientStop { position: 0.0; color: "#b75d71" }
                                GradientStop { position: 1.0; color: "#2c2634" }
                            }
                            Text {
                                anchors.centerIn: parent
                                text: launcher.hasGame ? "VI" : "VI"
                                color: "#f5cf8a"
                                font.pixelSize: 17
                                font.weight: Font.Bold
                            }
                        }
                        ColumnLayout {
                            Layout.fillWidth: true
                            spacing: 3
                            Text {
                                Layout.fillWidth: true
                                text: "Grand Theft Auto VI"
                                color: "#e6e2dc"
                                font.pixelSize: 12
                                font.weight: Font.DemiBold
                                elide: Text.ElideRight
                            }
                            Text {
                                text: launcher.gameExists ? "READY TO PLAY" : "ADD GAME FILES"
                                color: launcher.gameExists ? "#9ac7a4" : window.vicePink
                                font.pixelSize: 10
                                font.weight: Font.Bold
                                font.letterSpacing: 1.0
                            }
                        }
                        Text {
                            text: "···"
                            color: "#77797c"
                            font.pixelSize: 17
                        }
                    }
                }

                Item { Layout.fillHeight: true }

                Rectangle {
                    Layout.fillWidth: true
                    Layout.bottomMargin: 12
                    height: 68
                    radius: 10
                    color: "#1c1b24"
                    border.color: "#38303c"
                    Column {
                        anchors.fill: parent
                        anchors.leftMargin: 11
                        anchors.rightMargin: 8
                        anchors.topMargin: 10
                        spacing: 5
                        Text {
                            text: "WELCOME TO LEONIDA"
                            color: window.gold
                            font.pixelSize: 11
                            font.weight: Font.Bold
                            font.letterSpacing: 0.8
                        }
                        Text {
                            width: parent.width
                            text: "Vice City is waiting. Add your GTA VI game files."
                            color: "#8a8884"
                            font.pixelSize: 11
                            wrapMode: Text.WordWrap
                        }
                    }
                }

                Rectangle {
                    Layout.fillWidth: true
                    height: 39
                    radius: 10
                    color: addMouse.containsMouse ? "#f6d887" : window.gold
                    Row {
                        anchors.centerIn: parent
                        spacing: 8
                        Text {
                            text: "+"
                            color: "#211c18"
                            font.pixelSize: 16
                            font.weight: Font.DemiBold
                        }
                        Text {
                            text: "ADD GTA VI"
                            color: "#211c18"
                            font.pixelSize: 11
                            font.weight: Font.Bold
                            font.letterSpacing: 1.5
                        }
                    }
                    MouseArea {
                        id: addMouse
                        anchors.fill: parent
                        hoverEnabled: true
                        onClicked: launcher.browse()
                    }
                }

                Text {
                    Layout.topMargin: 15
                    Layout.leftMargin: 8
                    Layout.fillWidth: true
                    text: "ROCKSTAR GAMES  •  VICTORY FOR WINDOWS"
                    color: "#55585a"
                    font.pixelSize: 10
                    font.weight: Font.DemiBold
                    font.letterSpacing: 1
                    elide: Text.ElideRight
                }
            }
        }

        ColumnLayout {
            Layout.fillWidth: true
            Layout.fillHeight: true
            spacing: 0

            Rectangle {
                Layout.fillWidth: true
                Layout.preferredHeight: 72
                color: "#111315"
                border.color: "#292b2d"
                border.width: 1

                RowLayout {
                    anchors.fill: parent
                    anchors.leftMargin: 30
                    anchors.rightMargin: 28
                    spacing: 20

                    ColumnLayout {
                        spacing: 3
                        Text {
                            text: window.pageIndex === 0 ? "GRAND THEFT AUTO VI" : window.pageIndex === 1 ? "ROCKSTAR ACTIVITY" : "ROCKSTAR SETTINGS"
                            color: "#f0ede8"
                            font.pixelSize: 13
                            font.weight: Font.Bold
                            font.letterSpacing: 2.1
                        }
                        Text {
                            text: window.pageIndex === 0 ? "Vice City  •  State of Leonida" : window.pageIndex === 1 ? "Your local GTA VI launch history." : "Your GTA VI launcher preferences."
                            color: "#777a7c"
                            font.pixelSize: 11
                        }
                    }

                    Item { Layout.fillWidth: true }

                    Rectangle {
                        Layout.preferredWidth: 113
                        Layout.preferredHeight: 29
                        radius: 15
                        color: "#211d20"
                        border.color: "#493542"
                        Row {
                            anchors.centerIn: parent
                            spacing: 7
                            Rectangle {
                                width: 6
                                height: 6
                                radius: 3
                                anchors.verticalCenter: parent.verticalCenter
                                color: window.vicePink
                            }
                            Text {
                                text: "LOCAL LIBRARY"
                                color: "#e8c4a4"
                                font.pixelSize: 10
                                font.weight: Font.Bold
                                font.letterSpacing: 0.8
                            }
                        }
                    }
                    Rectangle {
                        width: 31
                        height: 31
                        radius: 16
                        color: "#282a2b"
                        border.color: "#464440"
                        Text {
                            anchors.centerIn: parent
                            text: "R*"
                            color: window.gold
                            font.pixelSize: 14
                            font.weight: Font.Bold
                        }
                    }
                }
            }

            StackLayout {
                id: pages
                Layout.fillWidth: true
                Layout.fillHeight: true
                currentIndex: window.pageIndex

                Item {
                    RowLayout {
                        anchors.fill: parent
                        anchors.margins: 25
                        spacing: 19

                        ColumnLayout {
                            Layout.fillWidth: true
                            Layout.fillHeight: true
                            spacing: 14

                            RowLayout {
                                Layout.fillWidth: true
                                Layout.bottomMargin: 1
                                Text {
                                    text: "ROCKSTAR GAMES PRESENTS"
                                    color: window.vicePink
                                    font.pixelSize: 11
                                    font.weight: Font.Bold
                                    font.letterSpacing: 1.9
                                }
                                Rectangle { Layout.fillWidth: true; height: 1; color: "#27292b" }
                                Text {
                                    text: "GTA VI  /  LEONIDA"
                                    color: "#67696b"
                                    font.pixelSize: 10
                                    font.letterSpacing: 1
                                }
                            }

                            Rectangle {
                                id: hero
                                Layout.fillWidth: true
                                Layout.fillHeight: true
                                Layout.minimumHeight: 330
                                radius: 14
                                clip: true
                                color: "#241d20"
                                border.color: "#493342"

                                Image {
                                    anchors.fill: parent
                                    source: launcher.wallpaperSource
                                    fillMode: Image.PreserveAspectCrop
                                    sourceSize.width: 3840
                                    sourceSize.height: 2160
                                    asynchronous: true
                                    smooth: true
                                }
                                Rectangle {
                                    anchors.fill: parent
                                    gradient: Gradient {
                                        GradientStop { position: 0.0; color: "#20151924" }
                                        GradientStop { position: 0.42; color: "#50171925" }
                                        GradientStop { position: 1.0; color: "#ed0b0d12" }
                                    }
                                }
                                Rectangle {
                                    anchors.fill: parent
                                    gradient: Gradient {
                                        orientation: Gradient.Horizontal
                                        GradientStop { position: 0.0; color: "#c0101017" }
                                        GradientStop { position: 0.58; color: "#6010131c" }
                                        GradientStop { position: 1.0; color: "#25101420" }
                                    }
                                }

                                RowLayout {
                                    anchors.left: parent.left
                                    anchors.top: parent.top
                                    anchors.margins: 24
                                    spacing: 8
                                    Rectangle {
                                        width: 24
                                        height: 24
                                        radius: 12
                                        color: window.gold
                                        Text {
                                            anchors.centerIn: parent
                                            text: "R*"
                                            color: "#211d19"
                                            font.pixelSize: 14
                                            font.weight: Font.Black
                                        }
                                    }
                                    Text {
                                        text: "ROCKSTAR GAMES"
                                        color: "#f2e9dc"
                                        font.pixelSize: 10
                                        font.weight: Font.Bold
                                        font.letterSpacing: 1.6
                                    }
                                    Rectangle { width: 1; height: 12; color: "#d5c7b077" }
                                    Text {
                                        text: "GRAND THEFT AUTO VI"
                                        color: "#d8c8b2"
                                        font.pixelSize: 10
                                        font.weight: Font.DemiBold
                                        font.letterSpacing: 1.1
                                    }
                                }

                                Rectangle {
                                    anchors.right: parent.right
                                    anchors.top: parent.top
                                    anchors.margins: 18
                                    width: 34
                                    height: 34
                                    radius: 9
                                    color: "#10121466"
                                    border.color: "#22ffffff"
                                    Text {
                                        anchors.centerIn: parent
                                        text: "···"
                                        color: "#eee8de"
                                        font.pixelSize: 17
                                    }
                                }

                                ColumnLayout {
                                    anchors.left: parent.left
                                    anchors.bottom: parent.bottom
                                    anchors.leftMargin: 30
                                    anchors.bottomMargin: 27
                                    width: Math.min(parent.width - 60, 560)
                                    spacing: 8

                                    Text {
                                        text: launcher.hasGame ? "GTA VI  /  READY TO PLAY" : "ROCKSTAR GAMES  /  VICE CITY"
                                        color: window.gold
                                        font.pixelSize: 11
                                        font.weight: Font.Bold
                                        font.letterSpacing: 2.0
                                    }
                                    Text {
                                        Layout.fillWidth: true
                                        text: "GRAND THEFT AUTO VI"
                                        color: "#f5f2ec"
                                        font.family: "Segoe UI Semibold"
                                        font.pixelSize: launcher.hasGame ? 36 : 43
                                        font.weight: Font.DemiBold
                                        font.letterSpacing: 0.7
                                        elide: Text.ElideRight
                                    }
                                    Text {
                                        Layout.fillWidth: true
                                        text: launcher.hasGame ? launcher.selectedPath : "Welcome to Leonida. Jason and Lucia's story begins in Vice City."
                                        color: "#e2dbe0"
                                        font.pixelSize: 12
                                        elide: Text.ElideMiddle
                                    }

                                    RowLayout {
                                        Layout.topMargin: 8
                                        spacing: 13
                                        Rectangle {
                                            Layout.preferredWidth: 158
                                            Layout.preferredHeight: 46
                                            radius: 10
                                            color: playMouse.containsMouse && !launcher.running ? "#f7df9b" : window.gold
                                            opacity: launcher.running ? 0.52 : 1
                                            Row {
                                                anchors.centerIn: parent
                                                spacing: 10
                                                Text {
                                                    text: launcher.running ? "●" : "▶"
                                                    color: "#211e1b"
                                                    font.pixelSize: 12
                                                }
                                                Text {
                                                    text: launcher.running ? "IN SESSION" : launcher.gameExists ? "PLAY GTA VI" : launcher.hasGame ? "LOCATE GTA VI" : "ADD GAME FILES"
                                                    color: "#211e1b"
                                                    font.pixelSize: 11
                                                    font.weight: Font.Bold
                                                    font.letterSpacing: 1.4
                                                }
                                            }
                                            MouseArea {
                                                id: playMouse
                                                anchors.fill: parent
                                                hoverEnabled: true
                                                enabled: !launcher.running
                                                onClicked: launcher.gameExists ? launcher.launch() : launcher.browse()
                                            }
                                        }
                                        Text {
                                            text: launcher.gameExists ? "GTA VI  •  " + launcher.consoleModel : launcher.status
                                            color: "#e2d6c6"
                                            font.pixelSize: 11
                                            font.letterSpacing: 0.3
                                        }
                                    }
                                }

                                Row {
                                    anchors.right: parent.right
                                    anchors.bottom: parent.bottom
                                    anchors.rightMargin: 22
                                    anchors.bottomMargin: 29
                                    spacing: 8
                                    Rectangle { width: 27; height: 2; color: window.gold }
                                    Rectangle { width: 15; height: 2; color: "#efe6d866" }
                                    Rectangle { width: 15; height: 2; color: "#efe6d866" }
                                }
                            }

                            Rectangle {
                                Layout.fillWidth: true
                                Layout.preferredHeight: 184
                                radius: 12
                                color: "#151719"
                                border.color: "#292b2d"
                                ColumnLayout {
                                    anchors.fill: parent
                                    spacing: 0
                                    RowLayout {
                                        Layout.fillWidth: true
                                        Layout.preferredHeight: 39
                                        Layout.leftMargin: 15
                                        Layout.rightMargin: 13
                                        Text {
                                            text: "ROCKSTAR LAUNCH ACTIVITY"
                                            color: "#d4d0ca"
                                            font.pixelSize: 10
                                            font.weight: Font.Bold
                                            font.letterSpacing: 1.5
                                        }
                                        Item { Layout.fillWidth: true }
                                        Rectangle {
                                            width: 6
                                            height: 6
                                            radius: 9
                                            color: launcher.running ? window.vicePink : window.ocean
                                        }
                                        Text {
                                            text: launcher.running ? "GTA VI LIVE" : "LOCAL LOG"
                                            color: launcher.running ? window.vicePink : window.ocean
                                            font.pixelSize: 10
                                            font.weight: Font.Bold
                                            font.letterSpacing: 1.0
                                        }
                                        Rectangle {
                                            Layout.leftMargin: 11
                                            width: 34
                                            height: 30
                                            radius: 9
                                            color: clearMouse.containsMouse ? "#2b2d2f" : "transparent"
                                            Text {
                                                anchors.centerIn: parent
                                                text: "↺"
                                                color: "#929393"
                                                font.pixelSize: 14
                                            }
                                            MouseArea {
                                                id: clearMouse
                                                anchors.fill: parent
                                                hoverEnabled: true
                                                onClicked: launcher.clearActivity()
                                            }
                                        }
                                    }
                                    Rectangle { Layout.fillWidth: true; height: 1; color: "#282a2c" }
                                    ListView {
                                        id: activityPreview
                                        Layout.fillWidth: true
                                        Layout.fillHeight: true
                                        Layout.margins: 12
                                        clip: true
                                        spacing: 7
                                        model: launcher.logs
                                        onCountChanged: positionViewAtEnd()
                                        delegate: RowLayout {
                                            required property string modelData
                                            required property int index
                                            width: activityPreview.width
                                            spacing: 9
                                            Rectangle {
                                                width: 5
                                                height: 5
                                                radius: 3
                                                color: index === activityPreview.count - 1 ? window.gold : "#565a5b"
                                            }
                                            Text {
                                                Layout.fillWidth: true
                                                text: modelData
                                                color: "#a7a6a2"
                                                font.family: "Cascadia Mono"
                                                font.pixelSize: 11
                                                elide: Text.ElideRight
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        ColumnLayout {
                            Layout.preferredWidth: 276
                            Layout.minimumWidth: 242
                            Layout.fillHeight: true
                            spacing: 12

                            Rectangle {
                                Layout.fillWidth: true
                                Layout.preferredHeight: 190
                                radius: 12
                                color: "#17191b"
                                border.color: "#2b2d2f"
                                ColumnLayout {
                                    anchors.fill: parent
                                    anchors.margins: 16
                                    spacing: 12
                                    RowLayout {
                                        Layout.fillWidth: true
                                        Text {
                                            text: "GTA VI  /  CONSOLE PROFILE"
                                            color: "#e1d6da"
                                            font.pixelSize: 10
                                            font.weight: Font.Bold
                                            font.letterSpacing: 1.4
                                        }
                                        Item { Layout.fillWidth: true }
                                        Text { text: "ⓘ"; color: "#777a7c"; font.pixelSize: 14 }
                                    }
                                    Text {
                                        Layout.fillWidth: true
                                        text: "Choose the target Xbox profile for your Grand Theft Auto VI session."
                                        color: "#898b8d"
                                        font.pixelSize: 11
                                        wrapMode: Text.WordWrap
                                    }
                                    RowLayout {
                                        Layout.fillWidth: true
                                        Layout.topMargin: 2
                                        spacing: 7
                                        Repeater {
                                            model: ["Series X", "Series S"]
                                            delegate: Rectangle {
                                                required property string modelData
                                                Layout.fillWidth: true
                                                Layout.preferredHeight: 43
                                                radius: 9
                                                color: launcher.consoleModel === modelData ? "#332a21" : "#202224"
                                                border.color: launcher.consoleModel === modelData ? "#805b38" : "#303234"
                                                Column {
                                                    anchors.centerIn: parent
                                                    spacing: 2
                                                    Text {
                                                        anchors.horizontalCenter: parent.horizontalCenter
                                                        text: modelData
                                                        color: launcher.consoleModel === modelData ? "#e9c18f" : "#b1b1af"
                                                        font.pixelSize: 11
                                                        font.weight: Font.DemiBold
                                                    }
                                                    Text {
                                                        anchors.horizontalCenter: parent.horizontalCenter
                                                        text: modelData === "Series X" ? "PERFORMANCE" : "EFFICIENCY"
                                                        color: "#77797b"
                                                        font.pixelSize: 10
                                                        font.weight: Font.Bold
                                                        font.letterSpacing: 0.8
                                                    }
                                                }
                                                MouseArea {
                                                    anchors.fill: parent
                                                    onClicked: launcher.setConsoleModel(modelData)
                                                }
                                            }
                                        }
                                    }
                                    Item { Layout.fillHeight: true }
                                    RowLayout {
                                        Layout.fillWidth: true
                                        Text {
                                            text: "GTA VI PROFILE"
                                            color: "#6d7072"
                                            font.pixelSize: 11
                                            font.weight: Font.Bold
                                            font.letterSpacing: 1.1
                                        }
                                        Item { Layout.fillWidth: true }
                                        Text {
                                            text: launcher.consoleModel.toUpperCase()
                                            color: window.gold
                                            font.pixelSize: 10
                                            font.weight: Font.Bold
                                            font.letterSpacing: 0.8
                                        }
                                    }
                                }
                            }

                            Rectangle {
                                Layout.fillWidth: true
                                Layout.preferredHeight: 181
                                radius: 12
                                color: "#17191b"
                                border.color: "#2b2d2f"
                                ColumnLayout {
                                    anchors.fill: parent
                                    anchors.margins: 16
                                    spacing: 10
                                    RowLayout {
                                        Layout.fillWidth: true
                                        Text {
                                            text: "GRAND THEFT AUTO VI FILES"
                                            color: "#e1d6da"
                                            font.pixelSize: 10
                                            font.weight: Font.Bold
                                            font.letterSpacing: 1.4
                                        }
                                        Item { Layout.fillWidth: true }
                                        Text {
                                            text: launcher.gameExists ? "●" : "○"
                                            color: launcher.gameExists ? "#81a987" : "#8a7a69"
                                            font.pixelSize: 12
                                        }
                                    }
                                    Text {
                                        Layout.fillWidth: true
                                        Layout.fillHeight: true
                                        text: launcher.hasGame ? launcher.selectedPath : "Choose the GTA VI executable from its existing game folder."
                                        color: launcher.hasGame ? "#aaa8a3" : "#747779"
                                        font.pixelSize: 11
                                        wrapMode: Text.WrapAnywhere
                                        maximumLineCount: 4
                                        elide: Text.ElideMiddle
                                    }
                                    Rectangle {
                                        Layout.fillWidth: true
                                        Layout.preferredHeight: 36
                                        radius: 9
                                        color: browseMouse.containsMouse ? "#292a2b" : "#202224"
                                        border.color: "#343638"
                                        Text {
                                            anchors.centerIn: parent
                                            text: launcher.hasGame ? "CHANGE GTA VI LOCATION" : "SELECT GTA VI EXECUTABLE"
                                            color: window.gold
                                            font.pixelSize: 10
                                            font.weight: Font.Bold
                                            font.letterSpacing: 1.0
                                        }
                                        MouseArea {
                                            id: browseMouse
                                            anchors.fill: parent
                                            hoverEnabled: true
                                            onClicked: launcher.browse()
                                        }
                                    }
                                }
                            }

                            Rectangle {
                                Layout.fillWidth: true
                                Layout.fillHeight: true
                                Layout.minimumHeight: 130
                                radius: 12
                                color: "#17191b"
                                border.color: "#2b2d2f"
                                ColumnLayout {
                                    anchors.fill: parent
                                    anchors.margins: 16
                                    spacing: 10
                                    Text {
                                        text: "ROCKSTAR LAUNCH STATUS"
                                        color: "#e1d6da"
                                        font.pixelSize: 10
                                        font.weight: Font.Bold
                                        font.letterSpacing: 1.4
                                    }
                                    RowLayout {
                                        Layout.fillWidth: true
                                        spacing: 9
                                        Rectangle {
                                            width: 30
                                            height: 30
                                            radius: 15
                                            color: launcher.running ? "#3a2c20" : launcher.gameExists ? "#202b22" : "#272421"
                                            Text {
                                                anchors.centerIn: parent
                                                text: launcher.running ? "▶" : launcher.gameExists ? "✓" : "·"
                                                color: launcher.running ? "#e4b27d" : launcher.gameExists ? "#87aa8d" : "#a18b72"
                                                font.pixelSize: 14
                                                font.weight: Font.Bold
                                            }
                                        }
                                        ColumnLayout {
                                            Layout.fillWidth: true
                                            spacing: 3
                                            Text {
                                                Layout.fillWidth: true
                                                text: launcher.status
                                                color: "#e5e1db"
                                                font.pixelSize: 11
                                                font.weight: Font.DemiBold
                                                elide: Text.ElideRight
                                            }
                                            Text {
                                                Layout.fillWidth: true
                                                text: launcher.running ? "Your GTA VI session is active." : launcher.gameExists ? "GTA VI is ready for your next session." : "Add the GTA VI executable to your library."
                                                color: "#858789"
                                                font.pixelSize: 10
                                                elide: Text.ElideRight
                                            }
                                        }
                                    }
                                    Rectangle { Layout.fillWidth: true; height: 1; color: "#2b2d2f" }
                                    Text {
                                        Layout.fillWidth: true
                                        text: "Your Rockstar game files stay in place. VIctory stores only launcher settings and local activity."
                                        color: "#757779"
                                        font.pixelSize: 10
                                        lineHeight: 1.25
                                        wrapMode: Text.WordWrap
                                    }
                                    Item { Layout.fillHeight: true }
                                }
                            }
                        }
                    }
                }

                Item {
                    ColumnLayout {
                        anchors.fill: parent
                        anchors.margins: 30
                        spacing: 14
                        RowLayout {
                            Layout.fillWidth: true
                            ColumnLayout {
                                spacing: 4
                                Text {
                                    text: "ROCKSTAR SESSION ACTIVITY"
                                    color: "#f0ede8"
                                    font.pixelSize: 14
                                    font.weight: Font.Bold
                                    font.letterSpacing: 1.2
                                }
                                Text {
                                    text: "GTA VI launcher and game output stays on this PC."
                                    color: "#85888a"
                                    font.pixelSize: 12
                                }
                            }
                            Item { Layout.fillWidth: true }
                            Rectangle {
                                width: 112
                                height: 32
                                radius: 9
                                color: clearActivityMouse.containsMouse ? "#292b2d" : "#1b1d1f"
                                border.color: "#353739"
                                Text {
                                    anchors.centerIn: parent
                                    text: "CLEAR ACTIVITY"
                                    color: "#c8c3bb"
                                    font.pixelSize: 10
                                    font.weight: Font.Bold
                                    font.letterSpacing: 1
                                }
                                MouseArea {
                                    id: clearActivityMouse
                                    anchors.fill: parent
                                    hoverEnabled: true
                                    onClicked: launcher.clearActivity()
                                }
                            }
                        }
                        Rectangle {
                            Layout.fillWidth: true
                            Layout.fillHeight: true
                            radius: 12
                            color: "#141618"
                            border.color: "#2b2d2f"
                            ColumnLayout {
                                anchors.fill: parent
                                spacing: 0
                                RowLayout {
                                    Layout.fillWidth: true
                                    Layout.preferredHeight: 43
                                    Layout.leftMargin: 17
                                    Layout.rightMargin: 17
                                    Text {
                                        text: "TIMESTAMP"
                                        color: "#6d7072"
                                        font.pixelSize: 10
                                        font.weight: Font.Bold
                                        font.letterSpacing: 1.4
                                    }
                                    Item { Layout.fillWidth: true }
                                    Text {
                                        text: launcher.running ? "GTA VI LIVE" : "LOCAL LOG"
                                        color: launcher.running ? window.vicePink : window.ocean
                                        font.pixelSize: 10
                                        font.weight: Font.Bold
                                        font.letterSpacing: 1.2
                                    }
                                }
                                Rectangle { Layout.fillWidth: true; height: 1; color: "#292b2d" }
                                ListView {
                                    id: fullActivity
                                    Layout.fillWidth: true
                                    Layout.fillHeight: true
                                    Layout.margins: 17
                                    clip: true
                                    spacing: 9
                                    model: launcher.logs
                                    onCountChanged: positionViewAtEnd()
                                    delegate: RowLayout {
                                        required property string modelData
                                        required property int index
                                        width: fullActivity.width
                                        spacing: 12
                                        Rectangle {
                                            width: 6
                                            height: 6
                                            radius: 3
                                            Layout.alignment: Qt.AlignVCenter
                                            color: index === fullActivity.count - 1 ? window.gold : "#5d6261"
                                        }
                                        Text {
                                            Layout.fillWidth: true
                                            text: modelData
                                            color: "#b6b3ae"
                                            font.family: "Cascadia Mono"
                                            font.pixelSize: 12
                                            wrapMode: Text.Wrap
                                        }
                                    }
                                    Text {
                                        anchors.centerIn: parent
                                        visible: fullActivity.count === 0
                                        text: "Your Rockstar launch activity will appear here."
                                        color: "#696c6e"
                                        font.pixelSize: 13
                                    }
                                }
                            }
                        }
                    }
                }

                Item {
                    ColumnLayout {
                        anchors.fill: parent
                        anchors.margins: 30
                        spacing: 14
                        Text {
                            text: "ROCKSTAR  /  GTA VI PREFERENCES"
                            color: "#f0ede8"
                            font.pixelSize: 14
                            font.weight: Font.Bold
                            font.letterSpacing: 1.2
                        }
                        Text {
                            text: "Grand Theft Auto VI launcher settings and local data."
                            color: "#85888a"
                            font.pixelSize: 12
                        }
                        Rectangle {
                            Layout.fillWidth: true
                            Layout.preferredHeight: 112
                            radius: 12
                            color: "#17191b"
                            border.color: "#2b2d2f"
                            ColumnLayout {
                                anchors.fill: parent
                                anchors.margins: 18
                                spacing: 7
                                Text {
                                    text: "GRAND THEFT AUTO VI CONSOLE PROFILE"
                                    color: window.gold
                                    font.pixelSize: 10
                                    font.weight: Font.Bold
                                    font.letterSpacing: 1.3
                                }
                                Text {
                                    text: "New sessions will use " + launcher.consoleModel + "."
                                    color: "#e2dfd9"
                                    font.pixelSize: 13
                                }
                                Text {
                                    text: "Change this Rockstar profile on the GTA VI library page."
                                    color: "#85888a"
                                    font.pixelSize: 11
                                }
                            }
                        }
                        Rectangle {
                            Layout.fillWidth: true
                            Layout.preferredHeight: 132
                            radius: 12
                            color: "#17191b"
                            border.color: "#2b2d2f"
                            ColumnLayout {
                                anchors.fill: parent
                                anchors.margins: 18
                                spacing: 8
                                Text {
                                    text: "ROCKSTAR LAUNCHER DATA"
                                    color: window.gold
                                    font.pixelSize: 10
                                    font.weight: Font.Bold
                                    font.letterSpacing: 1.3
                                }
                                Text {
                                    Layout.fillWidth: true
                                    text: launcher.dataDirectory
                                    color: "#c2bfba"
                                    font.family: "Cascadia Mono"
                                    font.pixelSize: 11
                                    wrapMode: Text.Wrap
                                }
                                Text {
                                    text: "Includes the GTA VI executable path, console profile, and launch activity."
                                    color: "#85888a"
                                    font.pixelSize: 11
                                }
                            }
                        }
                        Rectangle {
                            Layout.fillWidth: true
                            Layout.preferredHeight: 98
                            radius: 12
                            color: "#17191b"
                            border.color: "#2b2d2f"
                            RowLayout {
                                anchors.fill: parent
                                anchors.margins: 18
                                spacing: 14
                                Rectangle {
                                    width: 42
                                    height: 42
                                    radius: 9
                                    color: window.gold
                                    Text {
                                        anchors.centerIn: parent
                                        text: "R*"
                                        color: "#211d19"
                                        font.pixelSize: 22
                                        font.weight: Font.Black
                                    }
                                }
                                ColumnLayout {
                                    Layout.fillWidth: true
                                    spacing: 4
                                    Text {
                                        text: "ROCKSTAR GAMES  /  GTA VI"
                                        color: "#e9e5de"
                                        font.pixelSize: 12
                                        font.weight: Font.Bold
                                        font.letterSpacing: 1.2
                                    }
                                    Text {
                                        text: "VIctory desktop launcher  •  Python 3.13  •  PySide6 / QML"
                                        color: "#87898b"
                                        font.pixelSize: 11
                                    }
                                }
                            }
                        }
                        Item { Layout.fillHeight: true }
                    }
                }
            }
        }
    }
}
